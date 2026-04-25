from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationPose:
    name: str
    x: float
    y: float = 0.0
    z: float = 0.85


@dataclass(frozen=True)
class MotionSegment:
    frame: int
    action: str
    station: str
    phase: str
    gesture: str
    root_xyz: tuple[float, float, float]
    left_hand_xyz: tuple[float, float, float]
    right_hand_xyz: tuple[float, float, float]
    caption: str
    status: str = "success"
    progress_fraction: float = 0.0
    tool_call_index: int = 0


STATION_POSES: dict[str, StationPose] = {
    "pantry": StationPose("pantry", -1.7),
    "counter": StationPose("counter", -0.85),
    "sink": StationPose("sink", 0.0),
    "table": StationPose("table", 0.85),
    "delivery": StationPose("delivery", 1.7),
}

PHASE_GESTURES = {
    "scan": "scan",
    "navigate": "walk",
    "grasp": "grasp",
    "place": "place",
    "verify": "inspect",
    "report": "handoff",
}


def station_position(station: str) -> StationPose:
    return STATION_POSES.get(station, STATION_POSES["counter"])


def gesture_for_phase(phase: str) -> str:
    return PHASE_GESTURES.get(phase, "work")


def status_for_event(event: dict[str, object]) -> str:
    action = str(event.get("action") or "")
    rationale = str(event.get("rationale") or "").lower()
    reward = float(event.get("reward") or 0.0)
    if action == "wait" or reward < 0:
        return "failed"
    if "recovery" in rationale or "recovers" in rationale:
        return "recovery"
    return "success"


def gesture_for_status_phase(status: str, phase: str) -> str:
    if status == "failed":
        return "stalled"
    if status == "recovery":
        return "recover"
    return gesture_for_phase(phase)


def _hand_targets(pose: StationPose, gesture: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if gesture == "stalled":
        return (pose.x - 0.08, pose.y - 0.13, 0.98), (pose.x - 0.08, pose.y + 0.13, 0.98)
    if gesture == "recover":
        return (pose.x + 0.18, pose.y - 0.26, 1.16), (pose.x + 0.28, pose.y + 0.24, 1.10)
    if gesture == "grasp":
        return (pose.x + 0.12, pose.y - 0.22, 1.08), (pose.x + 0.20, pose.y + 0.24, 1.02)
    if gesture == "place":
        return (pose.x + 0.12, pose.y - 0.16, 1.02), (pose.x + 0.22, pose.y + 0.18, 0.98)
    if gesture == "inspect":
        return (pose.x + 0.08, pose.y - 0.18, 1.20), (pose.x + 0.08, pose.y + 0.18, 1.20)
    if gesture == "handoff":
        return (pose.x + 0.24, pose.y - 0.12, 1.10), (pose.x + 0.28, pose.y + 0.12, 1.10)
    return (pose.x, pose.y - 0.16, 1.12), (pose.x, pose.y + 0.16, 1.12)


def compile_humanoid_motion(events: list[dict[str, object]] | tuple[dict[str, object], ...]) -> list[MotionSegment]:
    segments: list[MotionSegment] = []
    delay_frames = 0
    tool_call_index = 0
    for event in events:
        if event.get("tool_name") != "execute_skill":
            continue
        tool_call_index += 1
        summary = event.get("observation_summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        action = str(event.get("action") or "work")
        expected = str(summary.get("expected_next") or action)
        station = str(summary.get("station") or "counter")
        phase = str(summary.get("phase") or action.split("_", 1)[0])
        frame = int(summary.get("frame_index") or 0) + delay_frames
        pose = station_position(station)
        status = status_for_event(event)
        gesture = gesture_for_status_phase(status, phase)
        left_hand, right_hand = _hand_targets(pose, gesture)
        caption = f"WAIT: {expected}" if status == "failed" else action
        segments.append(
            MotionSegment(
                frame=frame,
                action=action,
                station=pose.name,
                phase=phase,
                gesture=gesture,
                root_xyz=(pose.x, pose.y, pose.z),
                left_hand_xyz=left_hand,
                right_hand_xyz=right_hand,
                caption=caption,
                status=status,
                progress_fraction=float(summary.get("progress_fraction") or 0.0),
                tool_call_index=tool_call_index,
            )
        )
        if status == "failed":
            delay_frames += 18
        elif status == "recovery":
            delay_frames += 6
    return sorted(segments, key=lambda segment: segment.frame)


def summarize_tool_trace(events: list[dict[str, object]] | tuple[dict[str, object], ...]) -> dict[str, object]:
    stations: dict[str, int] = {}
    total = 0
    execute = 0
    score = 0
    failed = 0
    recovery = 0
    for event in events:
        tool_name = str(event.get("tool_name") or "")
        total += 1
        execute += 1 if tool_name == "execute_skill" else 0
        score += 1 if tool_name == "score_progress" else 0
        if tool_name == "execute_skill":
            status = status_for_event(event)
            failed += 1 if status == "failed" else 0
            recovery += 1 if status == "recovery" else 0
        summary = event.get("observation_summary") or {}
        if isinstance(summary, dict):
            station = str(summary.get("station") or "lab")
            stations[station] = stations.get(station, 0) + 1
    return {
        "total_tool_calls": total,
        "execute_skill_calls": execute,
        "score_progress_calls": score,
        "failed_execute_calls": failed,
        "recovery_execute_calls": recovery,
        "stations": stations,
    }


def lane_status_counts(segments: list[MotionSegment] | tuple[MotionSegment, ...]) -> dict[str, int]:
    counts = {"success": 0, "failed": 0, "recovery": 0}
    for segment in segments:
        counts[segment.status] = counts.get(segment.status, 0) + 1
    return counts
