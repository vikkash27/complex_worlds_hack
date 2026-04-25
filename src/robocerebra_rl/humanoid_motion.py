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


def _hand_targets(pose: StationPose, gesture: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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
    for event in events:
        if event.get("tool_name") != "execute_skill":
            continue
        summary = event.get("observation_summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        action = str(event.get("action") or "work")
        station = str(summary.get("station") or "counter")
        phase = str(summary.get("phase") or action.split("_", 1)[0])
        frame = int(summary.get("frame_index") or 0)
        pose = station_position(station)
        gesture = gesture_for_phase(phase)
        left_hand, right_hand = _hand_targets(pose, gesture)
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
                caption=action,
            )
        )
    return sorted(segments, key=lambda segment: segment.frame)


def summarize_tool_trace(events: list[dict[str, object]] | tuple[dict[str, object], ...]) -> dict[str, object]:
    stations: dict[str, int] = {}
    total = 0
    execute = 0
    score = 0
    for event in events:
        tool_name = str(event.get("tool_name") or "")
        total += 1
        execute += 1 if tool_name == "execute_skill" else 0
        score += 1 if tool_name == "score_progress" else 0
        summary = event.get("observation_summary") or {}
        if isinstance(summary, dict):
            station = str(summary.get("station") or "lab")
            stations[station] = stations.get(station, 0) + 1
    return {
        "total_tool_calls": total,
        "execute_skill_calls": execute,
        "score_progress_calls": score,
        "stations": stations,
    }
