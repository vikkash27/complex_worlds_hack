from __future__ import annotations

import math

import numpy as np

from robocerebra_rl.humanoid_motion import MotionSegment


def default_mujoco_g1_joint_names() -> tuple[str, ...]:
    return (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    )


def mujoco_joint_target_dict(segment: MotionSegment) -> dict[str, float]:
    step = math.sin(segment.frame / 8.0)
    walk_amp = 0.34 if segment.gesture == "walk" else 0.05
    reach = 0.75 if segment.gesture in {"grasp", "place", "handoff", "recover"} else 0.22
    elbow = 0.85 if segment.gesture in {"grasp", "place", "handoff", "recover"} else 0.18
    torso = -0.24 if segment.status == "failed" else 0.18 if segment.status == "recovery" else 0.0
    if segment.gesture == "stalled":
        walk_amp = 0.0
        reach = -0.1
        elbow = 0.1
    return {
        "left_hip_pitch_joint": -walk_amp * step,
        "right_hip_pitch_joint": walk_amp * step,
        "left_knee_joint": max(0.0, walk_amp * step) * 0.9,
        "right_knee_joint": max(0.0, -walk_amp * step) * 0.9,
        "left_ankle_pitch_joint": walk_amp * step * 0.25,
        "right_ankle_pitch_joint": -walk_amp * step * 0.25,
        "waist_yaw_joint": torso,
        "left_shoulder_pitch_joint": -reach * 0.35,
        "right_shoulder_pitch_joint": reach,
        "left_shoulder_roll_joint": 0.18 if segment.gesture in {"scan", "inspect"} else 0.05,
        "right_shoulder_roll_joint": -0.18 if segment.gesture in {"scan", "inspect"} else -0.05,
        "left_elbow_joint": elbow * 0.45,
        "right_elbow_joint": elbow,
        "left_wrist_roll_joint": -0.18 if segment.gesture in {"grasp", "place"} else 0.0,
        "right_wrist_roll_joint": 0.18 if segment.gesture in {"grasp", "place"} else 0.0,
    }


def build_mujoco_g1_qpos(segment: MotionSegment, joint_names: list[str] | tuple[str, ...]) -> np.ndarray:
    targets = mujoco_joint_target_dict(segment)
    return np.array([targets.get(name, 0.0) for name in joint_names], dtype=float)
