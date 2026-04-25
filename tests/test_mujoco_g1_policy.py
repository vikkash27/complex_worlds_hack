import numpy as np

from robocerebra_rl.humanoid_motion import MotionSegment
from robocerebra_rl.mujoco_g1_policy import (
    build_mujoco_g1_qpos,
    default_mujoco_g1_joint_names,
    mujoco_joint_target_dict,
)


def _segment(gesture: str, *, status: str = "success", frame: int = 24) -> MotionSegment:
    return MotionSegment(
        frame=frame,
        action=f"{gesture}_counter_1",
        station="counter",
        phase=gesture,
        gesture=gesture,
        root_xyz=(-0.85, 0.0, 0.85),
        left_hand_xyz=(-0.75, -0.18, 1.1),
        right_hand_xyz=(-0.65, 0.18, 1.1),
        caption=f"{gesture}_counter_1",
        status=status,
        progress_fraction=0.5,
        tool_call_index=3,
    )


def test_default_mujoco_g1_joint_names_include_legs_arms_and_waist():
    names = default_mujoco_g1_joint_names()

    assert "left_hip_pitch_joint" in names
    assert "right_knee_joint" in names
    assert "waist_yaw_joint" in names
    assert "right_elbow_joint" in names


def test_mujoco_joint_target_dict_differs_for_walk_and_grasp():
    walk = mujoco_joint_target_dict(_segment("walk"))
    grasp = mujoco_joint_target_dict(_segment("grasp"))

    assert walk["left_hip_pitch_joint"] == -walk["right_hip_pitch_joint"]
    assert grasp["right_shoulder_pitch_joint"] > walk["right_shoulder_pitch_joint"]
    assert grasp["right_elbow_joint"] > 0.2


def test_build_mujoco_g1_qpos_matches_joint_order_and_failed_pose():
    joint_names = ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "right_elbow_joint"]

    qpos = build_mujoco_g1_qpos(_segment("scan", status="failed"), joint_names)

    assert isinstance(qpos, np.ndarray)
    assert qpos.shape == (4,)
    assert qpos[2] < 0
