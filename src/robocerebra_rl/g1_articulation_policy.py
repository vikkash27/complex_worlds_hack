from __future__ import annotations

from dataclasses import dataclass
import math

from robocerebra_rl.humanoid_motion import MotionSegment


G1_GROUPS = ("left_leg", "right_leg", "left_arm", "right_arm", "torso")


@dataclass(frozen=True)
class G1JointTargets:
    frame: int
    gesture: str
    positions: dict[str, float]


@dataclass(frozen=True)
class G1LinkPose:
    link_name: str
    rotate_xyz: tuple[float, float, float]
    translate_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


def classify_g1_joint(name: str) -> str | None:
    lowered = name.lower()
    if "left" in lowered and any(part in lowered for part in ("hip", "knee", "ankle")):
        return "left_leg"
    if "right" in lowered and any(part in lowered for part in ("hip", "knee", "ankle")):
        return "right_leg"
    if "left" in lowered and any(part in lowered for part in ("shoulder", "elbow", "wrist")):
        return "left_arm"
    if "right" in lowered and any(part in lowered for part in ("shoulder", "elbow", "wrist")):
        return "right_arm"
    if any(part in lowered for part in ("waist", "torso", "spine")):
        return "torso"
    return None


def select_supported_g1_joints(joint_names: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    selected = {group: [] for group in G1_GROUPS}
    for name in joint_names:
        group = classify_g1_joint(name)
        if group:
            selected[group].append(name)
    return {group: names for group, names in selected.items() if names}


def _group_amplitude(gesture: str, group: str) -> float:
    if gesture == "walk":
        return 0.28 if group.endswith("leg") else 0.06
    if gesture == "grasp":
        return 0.42 if group == "right_arm" else 0.16 if group == "left_arm" else 0.04
    if gesture == "place":
        return 0.34 if group in {"left_arm", "right_arm"} else 0.04
    if gesture in {"inspect", "scan"}:
        return 0.18 if group in {"left_arm", "right_arm", "torso"} else 0.03
    if gesture == "handoff":
        return 0.30 if group in {"left_arm", "right_arm"} else 0.05
    return 0.08


def build_g1_policy_frame(segment: MotionSegment, supported_joints: dict[str, list[str]]) -> G1JointTargets:
    positions: dict[str, float] = {}
    step_phase = math.sin(segment.frame / 12.0)
    for group, names in supported_joints.items():
        amplitude = _group_amplitude(segment.gesture, group)
        sign = -1.0 if group in {"left_leg", "left_arm"} else 1.0
        if segment.gesture == "walk" and group in {"left_leg", "right_leg"}:
            value = sign * amplitude * step_phase
        else:
            value = sign * amplitude
        for name in names:
            positions[name] = round(value, 6)
    return G1JointTargets(frame=segment.frame, gesture=segment.gesture, positions=positions)


def build_g1_visual_link_poses(segment: MotionSegment) -> list[G1LinkPose]:
    phase = math.sin(segment.frame / 10.0)
    walk = 18.0 * phase if segment.gesture == "walk" else 0.0
    arm_reach = 28.0 if segment.gesture in {"grasp", "place", "handoff", "recover"} else 10.0 if segment.gesture in {"scan", "inspect"} else 0.0
    elbow = 34.0 if segment.gesture in {"grasp", "place", "handoff", "recover"} else 8.0
    torso_pitch = -10.0 if segment.status == "failed" else 8.0 if segment.gesture in {"grasp", "place"} else 0.0
    torso_drop = -0.045 if segment.status == "failed" else 0.0
    if segment.status == "recovery":
        torso_pitch = 12.0
        arm_reach = 34.0
    return [
        G1LinkPose("torso_link", (torso_pitch, 0.0, 0.0), (0.0, 0.0, torso_drop)),
        G1LinkPose("left_hip_pitch_link", (0.0, -walk, 0.0)),
        G1LinkPose("right_hip_pitch_link", (0.0, walk, 0.0)),
        G1LinkPose("left_knee_link", (0.0, max(0.0, walk) * 0.65, 0.0)),
        G1LinkPose("right_knee_link", (0.0, max(0.0, -walk) * 0.65, 0.0)),
        G1LinkPose("left_ankle_pitch_link", (0.0, -walk * 0.25, 0.0)),
        G1LinkPose("right_ankle_pitch_link", (0.0, walk * 0.25, 0.0)),
        G1LinkPose("left_shoulder_pitch_link", (0.0, -arm_reach * 0.45, 0.0)),
        G1LinkPose("right_shoulder_pitch_link", (0.0, arm_reach, 0.0)),
        G1LinkPose("left_shoulder_roll_link", (8.0 if segment.gesture in {"scan", "inspect"} else 0.0, 0.0, 0.0)),
        G1LinkPose("right_shoulder_roll_link", (-8.0 if segment.gesture in {"scan", "inspect"} else 0.0, 0.0, 0.0)),
        G1LinkPose("left_elbow_link", (0.0, elbow * 0.45, 0.0)),
        G1LinkPose("right_elbow_link", (0.0, elbow, 0.0)),
        G1LinkPose("left_wrist_roll_link", (0.0, 0.0, -12.0 if segment.gesture in {"grasp", "place"} else 0.0)),
        G1LinkPose("right_wrist_roll_link", (0.0, 0.0, 12.0 if segment.gesture in {"grasp", "place"} else 0.0)),
    ]
