from robocerebra_rl.g1_articulation_policy import (
    G1JointTargets,
    build_g1_policy_frame,
    classify_g1_joint,
    select_supported_g1_joints,
)
from robocerebra_rl.humanoid_motion import MotionSegment


def _segment(gesture: str, frame: int = 24) -> MotionSegment:
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
    )


def test_classify_g1_joint_names_from_common_unitree_assets():
    assert classify_g1_joint("left_hip_pitch_joint") == "left_leg"
    assert classify_g1_joint("right_knee_joint") == "right_leg"
    assert classify_g1_joint("left_shoulder_pitch_joint") == "left_arm"
    assert classify_g1_joint("right_elbow_joint") == "right_arm"
    assert classify_g1_joint("waist_yaw_joint") == "torso"
    assert classify_g1_joint("floating_base") is None


def test_select_supported_g1_joints_keeps_relevant_groups():
    joints = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_elbow_joint",
        "waist_yaw_joint",
        "camera_joint",
    ]

    selected = select_supported_g1_joints(joints)

    assert selected["left_leg"] == ["left_hip_pitch_joint"]
    assert selected["right_leg"] == ["right_hip_pitch_joint"]
    assert selected["left_arm"] == ["left_shoulder_pitch_joint"]
    assert selected["right_arm"] == ["right_elbow_joint"]
    assert selected["torso"] == ["waist_yaw_joint"]
    assert "camera_joint" not in sum(selected.values(), [])


def test_build_g1_policy_frame_produces_gesture_targets():
    selected = select_supported_g1_joints(
        [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "waist_yaw_joint",
        ]
    )

    grasp = build_g1_policy_frame(_segment("grasp"), selected)
    walk = build_g1_policy_frame(_segment("walk"), selected)

    assert isinstance(grasp, G1JointTargets)
    assert grasp.frame == 24
    assert grasp.positions["right_shoulder_pitch_joint"] > walk.positions["right_shoulder_pitch_joint"]
    assert walk.positions["left_hip_pitch_joint"] == -walk.positions["right_hip_pitch_joint"]
