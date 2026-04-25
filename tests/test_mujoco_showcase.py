import json
from pathlib import Path

from PIL import Image

from robocerebra_rl.mujoco_showcase import (
    build_showcase_timeline,
    render_storyboard_frame,
    summarize_showcase_pair,
)


def _event(action: str, station: str, frame: int, reward: float, run_id: str) -> dict[str, object]:
    return {
        "tool_name": "execute_skill",
        "action": action,
        "reward": reward,
        "rationale": "Recovery execution." if "recover" in action else "",
        "run_id": run_id,
        "observation_summary": {
            "station": station,
            "phase": action.split("_", 1)[0],
            "frame_index": frame,
            "progress_fraction": frame / 120,
        },
    }


def test_build_showcase_timeline_aligns_baseline_and_optimized_statuses():
    baseline = [_event("wait", "counter", 12, -0.05, "baseline"), _event("scan_counter_1", "counter", 18, 0.1, "baseline")]
    optimized = [_event("scan_counter_1", "counter", 12, 0.1, "trained")]

    timeline = build_showcase_timeline(baseline, optimized)

    assert timeline.baseline[0].status == "failed"
    assert timeline.baseline[1].frame > 18
    assert timeline.optimized[0].status == "success"
    assert timeline.max_frame >= timeline.baseline[-1].frame


def test_summarize_showcase_pair_reports_openreward_difference():
    summary = summarize_showcase_pair(
        [_event("wait", "counter", 12, -0.05, "baseline"), _event("scan_counter_1", "counter", 18, 0.1, "baseline")],
        [_event("scan_counter_1", "counter", 12, 0.1, "trained")],
    )

    assert summary["baseline_failed_execute_calls"] == 1
    assert summary["optimized_failed_execute_calls"] == 0
    assert summary["baseline_execute_skill_calls"] == 2
    assert summary["optimized_execute_skill_calls"] == 1


def test_render_storyboard_frame_returns_pillow_image():
    timeline = build_showcase_timeline(
        [_event("wait", "counter", 12, -0.05, "baseline")],
        [_event("scan_counter_1", "counter", 12, 0.1, "trained")],
    )

    image = render_storyboard_frame(timeline, frame_index=0, size=(640, 360))

    assert isinstance(image, Image.Image)
    assert image.size == (640, 360)


def test_showcase_summary_can_be_serialized(tmp_path):
    summary = summarize_showcase_pair([_event("wait", "counter", 12, -0.05, "baseline")], [])
    path = tmp_path / "summary.json"

    path.write_text(json.dumps(summary), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8"))["baseline_failed_execute_calls"] == 1
