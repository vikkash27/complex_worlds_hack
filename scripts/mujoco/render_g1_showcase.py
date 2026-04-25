from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.humanoid_motion import MotionSegment  # noqa: E402
from robocerebra_rl.mujoco_artifact_layout import apply_artifact_mocap  # noqa: E402
from robocerebra_rl.mujoco_pick_place import apply_task_tray_pick_place  # noqa: E402
from robocerebra_rl.mujoco_assets import load_menagerie_g1_manifest  # noqa: E402
from robocerebra_rl.mujoco_g1_policy import build_mujoco_g1_qpos, default_mujoco_g1_joint_names  # noqa: E402
from robocerebra_rl.mujoco_showcase import (  # noqa: E402
    ShowcaseTimeline,
    build_showcase_timeline,
    load_trace,
    render_storyboard_frame,
    summarize_showcase_pair,
)


DEFAULT_BASELINE = ROOT / "artifacts" / "traces" / "humanoid_baseline_long_horizon.jsonl"
DEFAULT_OPTIMIZED = ROOT / "artifacts" / "traces" / "humanoid_trained_long_horizon.jsonl"
DEFAULT_MANIFEST = ROOT / "artifacts" / "mujoco" / "vendor" / "menagerie_g1_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "mujoco"


def _write_gif(frames: list[Image.Image], path: Path, *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def _frame_indices(timeline: ShowcaseTimeline, frame_count: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    return [int(timeline.max_frame * idx / (frame_count - 1)) for idx in range(frame_count)]


def _set_named_qpos(model, data, mujoco, segment: MotionSegment) -> None:
    qpos = build_mujoco_g1_qpos(segment, default_mujoco_g1_joint_names())
    targets = dict(zip(default_mujoco_g1_joint_names(), qpos, strict=True))
    for joint_index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
        if not name or name not in targets:
            continue
        qpos_index = model.jnt_qposadr[joint_index]
        data.qpos[qpos_index] = targets[name]


def _ensure_mujoco_offscreen_buffer(model, panel_w: int, height: int) -> None:
    """Resize MJCF offscreen framebuffer so :class:`mujoco.Renderer` can use ``(panel_w, height)``.

    Menagerie models often ship with a 480px-tall offscreen buffer; the side-by-side
    default uses 720px-tall panels.
    """
    g = model.vis.global_  # type: ignore[attr-defined]
    g.offwidth = max(int(g.offwidth), int(panel_w))
    g.offheight = max(int(g.offheight), int(height))


def _active_segment(segments: tuple[MotionSegment, ...], frame_index: int) -> MotionSegment | None:
    active = None
    for segment in segments:
        if segment.frame <= frame_index:
            active = segment
        else:
            break
    return active or (segments[0] if segments else None)


def _render_mujoco_pair(timeline: ShowcaseTimeline, scene_path: Path, *, frame_indices: list[int], size: tuple[int, int]) -> list[Image.Image]:
    import mujoco  # type: ignore[import-not-found]

    width, height = size
    panel_w = width // 2
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _ensure_mujoco_offscreen_buffer(model, panel_w, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=panel_w)
    frames: list[Image.Image] = []
    for frame_index in frame_indices:
        panels: list[Image.Image] = []
        for label, segments in [
            ("Baseline: wait failures + recovery retries", timeline.baseline),
            ("Optimized: smooth tool-chain execution", timeline.optimized),
        ]:
            segment = _active_segment(segments, frame_index)
            mujoco.mj_resetData(model, data)
            if segment is not None:
                _set_named_qpos(model, data, mujoco, segment)
            mujoco.mj_forward(model, data)
            apply_task_tray_pick_place(mujoco, model, data, segment, frame_index=frame_index)
            apply_artifact_mocap(mujoco, model, data, segment, frame_index=frame_index)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            rgb = renderer.render()
            panel = Image.fromarray(rgb)
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel_w, 72), fill=(25, 32, 45))
            draw.text((14, 14), label, fill=(255, 255, 255))
            if segment is not None:
                draw.text((14, 40), f"{segment.status}: {segment.caption}", fill=(200, 210, 225))
                draw.text(
                    (14, 58),
                    f"pick+place: free tray+mug, weld relpose to wrist; mocap: spill, tote  @  {segment.station}",
                    fill=(150, 160, 180),
                )
            panels.append(panel)
        combined = Image.new("RGB", size, (238, 242, 245))
        combined.paste(panels[0], (0, 0))
        combined.paste(panels[1], (panel_w, 0))
        frames.append(combined)
    renderer.close()
    return frames


def render_showcase(
    *,
    baseline_trace: Path,
    optimized_trace: Path,
    manifest: Path,
    output_dir: Path,
    frame_count: int,
    fps: int,
    backend: str,
    size: tuple[int, int] = (1280, 720),
) -> dict[str, object]:
    baseline_events = load_trace(baseline_trace)
    optimized_events = load_trace(optimized_trace)
    timeline = build_showcase_timeline(baseline_events, optimized_events)
    frame_indices = _frame_indices(timeline, frame_count)
    selected_backend = backend
    scene_path: Path | None = None
    frames: list[Image.Image]
    if backend in {"auto", "mujoco"}:
        try:
            scene_path = load_menagerie_g1_manifest(manifest).ensure_task_showcase_scene()
            frames = _render_mujoco_pair(timeline, scene_path, frame_indices=frame_indices, size=size)
            selected_backend = "mujoco"
        except Exception as exc:
            if backend == "mujoco":
                raise
            print(f"[mujoco-showcase] falling back to storyboard renderer: {exc}", flush=True)
            selected_backend = "storyboard"
            frames = [render_storyboard_frame(timeline, frame_index=idx, size=size) for idx in frame_indices]
    else:
        frames = [render_storyboard_frame(timeline, frame_index=idx, size=size) for idx in frame_indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "g1_openreward_showcase.gif"
    _write_gif(frames, gif_path, fps=fps)
    summary = {
        **summarize_showcase_pair(baseline_events, optimized_events),
        "backend": selected_backend,
        "scene": str(scene_path) if scene_path else "",
        "gif": str(gif_path),
        "frames": len(frames),
        "fps": fps,
    }
    summary_path = output_dir / "g1_openreward_showcase_summary.json"
    summary_path.write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(f"[mujoco-showcase] wrote {gif_path}")
    print(f"[mujoco-showcase] wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render side-by-side Unitree G1 RoboCerebra/OpenReward MuJoCo showcase (offline GIF; see scripts/mujoco/view_dynamic_grasp.py for a live 3D viewer with mj_step).",
    )
    parser.add_argument("--baseline-trace", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--optimized-trace", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--backend", choices=("auto", "mujoco", "storyboard"), default="auto")
    args = parser.parse_args()
    render_showcase(
        baseline_trace=args.baseline_trace,
        optimized_trace=args.optimized_trace,
        manifest=args.manifest,
        output_dir=args.output_dir,
        frame_count=args.frames,
        fps=args.fps,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
