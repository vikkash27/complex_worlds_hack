from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from robocerebra_rl.humanoid_motion import MotionSegment, compile_humanoid_motion, summarize_tool_trace
from robocerebra_rl.mujoco_g1_policy import build_mujoco_g1_qpos, default_mujoco_g1_joint_names


@dataclass(frozen=True)
class ShowcaseTimeline:
    baseline: tuple[MotionSegment, ...]
    optimized: tuple[MotionSegment, ...]
    max_frame: int


def load_trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_showcase_timeline(
    baseline_events: list[dict[str, object]] | tuple[dict[str, object], ...],
    optimized_events: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> ShowcaseTimeline:
    baseline = tuple(compile_humanoid_motion(baseline_events))
    optimized = tuple(compile_humanoid_motion(optimized_events))
    max_frame = max([segment.frame for segment in (*baseline, *optimized)], default=0)
    return ShowcaseTimeline(baseline=baseline, optimized=optimized, max_frame=max_frame)


def summarize_showcase_pair(
    baseline_events: list[dict[str, object]] | tuple[dict[str, object], ...],
    optimized_events: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> dict[str, object]:
    baseline = summarize_tool_trace(baseline_events)
    optimized = summarize_tool_trace(optimized_events)
    return {
        "baseline_execute_skill_calls": baseline["execute_skill_calls"],
        "optimized_execute_skill_calls": optimized["execute_skill_calls"],
        "baseline_failed_execute_calls": baseline["failed_execute_calls"],
        "optimized_failed_execute_calls": optimized["failed_execute_calls"],
        "baseline_recovery_execute_calls": baseline["recovery_execute_calls"],
        "optimized_recovery_execute_calls": optimized["recovery_execute_calls"],
        "baseline_total_tool_calls": baseline["total_tool_calls"],
        "optimized_total_tool_calls": optimized["total_tool_calls"],
    }


def _active_segment(segments: tuple[MotionSegment, ...], frame_index: int) -> MotionSegment | None:
    active: MotionSegment | None = None
    for segment in segments:
        if segment.frame <= frame_index:
            active = segment
        else:
            break
    return active or (segments[0] if segments else None)


def _status_color(status: str) -> tuple[int, int, int]:
    if status == "failed":
        return (230, 45, 35)
    if status == "recovery":
        return (245, 170, 35)
    return (45, 130, 245)


def _draw_robot(draw: ImageDraw.ImageDraw, center: tuple[int, int], segment: MotionSegment | None, label: str) -> None:
    x, y = center
    status = segment.status if segment else "success"
    color = _status_color(status)
    phase = segment.gesture if segment else "idle"
    qpos = build_mujoco_g1_qpos(segment, default_mujoco_g1_joint_names()) if segment else []
    stride = float(qpos[0]) if len(qpos) else 0.0
    reach = float(qpos[18]) if len(qpos) > 18 else 0.0
    arm_dx = int(38 * reach)
    leg_dx = int(35 * stride)
    draw.ellipse((x - 18, y - 95, x + 18, y - 59), fill=(235, 235, 235), outline=color, width=4)
    draw.line((x, y - 58, x, y + 20), fill=color, width=8)
    draw.line((x, y - 35, x - 42, y - 5), fill=color, width=7)
    draw.line((x, y - 35, x + 35 + arm_dx, y - 8), fill=color, width=7)
    draw.line((x, y + 20, x - 18 - leg_dx, y + 88), fill=color, width=8)
    draw.line((x, y + 20, x + 18 + leg_dx, y + 88), fill=color, width=8)
    draw.rectangle((x + 33 + arm_dx, y - 18, x + 58 + arm_dx, y + 6), fill=color)
    draw.text((x - 115, y + 105), label, fill=(15, 18, 22))
    if segment:
        draw.text((x - 115, y + 128), f"{phase} | {segment.caption}", fill=(15, 18, 22))


def _draw_lane(draw: ImageDraw.ImageDraw, y: int, title: str, segments: tuple[MotionSegment, ...], frame_index: int, width: int) -> None:
    draw.text((24, y - 145), title, fill=(15, 18, 22))
    draw.line((48, y + 112, width - 48, y + 112), fill=(190, 195, 205), width=5)
    for idx, segment in enumerate(segments):
        sx = 48 + int((width - 96) * min(1.0, max(0.0, segment.progress_fraction)))
        fill = _status_color(segment.status)
        draw.rectangle((sx - 5, y + 103, sx + 5, y + 121), fill=fill)
        if idx % 6 == 0:
            draw.text((sx - 18, y + 126), str(idx + 1), fill=(60, 65, 75))
    active = _active_segment(segments, frame_index)
    progress = active.progress_fraction if active else 0.0
    x = 100 + int((width - 220) * progress)
    _draw_robot(draw, (x, y), active, title)


def render_storyboard_frame(timeline: ShowcaseTimeline, *, frame_index: int, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    image = Image.new("RGB", size, (238, 242, 245))
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle((0, 0, width, 56), fill=(25, 32, 45))
    draw.text((24, 18), "RoboCerebra / OpenReward MuJoCo G1: baseline vs optimized", fill=(255, 255, 255))
    _draw_lane(draw, height // 3, "Baseline: wait failures + recovery retries", timeline.baseline, frame_index, width)
    _draw_lane(draw, 2 * height // 3, "Optimized: smooth tool-chain execution", timeline.optimized, frame_index, width)
    return image
