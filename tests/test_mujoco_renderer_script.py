import json

import pytest

from scripts.mujoco.render_g1_showcase import _ensure_mujoco_offscreen_buffer, render_showcase


def _event(action: str, frame: int, reward: float, run_id: str) -> dict[str, object]:
    return {
        "tool_name": "execute_skill",
        "action": action,
        "reward": reward,
        "run_id": run_id,
        "observation_summary": {
            "station": "counter",
            "phase": action.split("_", 1)[0],
            "frame_index": frame,
            "progress_fraction": frame / 48,
        },
    }


def test_render_showcase_storyboard_writes_gif_and_summary(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    optimized = tmp_path / "optimized.jsonl"
    baseline.write_text(json.dumps(_event("wait", 12, -0.05, "baseline")) + "\n", encoding="utf-8")
    optimized.write_text(json.dumps(_event("scan_counter_1", 12, 0.1, "trained")) + "\n", encoding="utf-8")

    summary = render_showcase(
        baseline_trace=baseline,
        optimized_trace=optimized,
        manifest=tmp_path / "missing_manifest.json",
        output_dir=tmp_path,
        frame_count=3,
        fps=4,
        backend="storyboard",
        size=(320, 180),
    )

    assert summary["backend"] == "storyboard"
    assert summary["baseline_failed_execute_calls"] == 1
    assert (tmp_path / "g1_openreward_showcase.gif").is_file()
    assert (tmp_path / "g1_openreward_showcase_summary.json").is_file()


def test_ensure_mujoco_offscreen_buffer_allows_720p_renderer():
    pytest.importorskip("mujoco")
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>"
    )
    _ensure_mujoco_offscreen_buffer(model, panel_w=640, height=720)
    # Would raise if framebuffer smaller than 640x720
    r = mujoco.Renderer(model, height=720, width=640)
    r.close()
