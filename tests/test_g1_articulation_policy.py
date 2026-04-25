from robocerebra_rl.g1_articulation_policy import (
    G1JointTargets,
    G1LinkPose,
    build_g1_policy_frame,
    build_g1_visual_link_poses,
    classify_g1_joint,
    g1_link_path_candidates,
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


def test_build_g1_visual_link_poses_animates_unitree_link_names():
    poses = build_g1_visual_link_poses(_segment("walk", frame=36))
    by_name = {pose.link_name: pose for pose in poses}

    assert isinstance(poses[0], G1LinkPose)
    assert "left_hip_pitch_link" in by_name
    assert "right_hip_pitch_link" in by_name
    assert "left_shoulder_pitch_link" in by_name
    assert "right_elbow_link" in by_name
    assert by_name["left_hip_pitch_link"].rotate_xyz != by_name["right_hip_pitch_link"].rotate_xyz


def test_build_g1_visual_link_poses_uses_arms_for_grasp_and_stall_for_failed_status():
    grasp = {pose.link_name: pose for pose in build_g1_visual_link_poses(_segment("grasp"))}
    stalled_segment = MotionSegment(
        frame=24,
        action="wait",
        station="counter",
        phase="scan",
        gesture="stalled",
        root_xyz=(-0.85, 0.0, 0.85),
        left_hand_xyz=(-0.9, -0.18, 1.0),
        right_hand_xyz=(-0.9, 0.18, 1.0),
        caption="WAIT",
        status="failed",
    )
    stalled = {pose.link_name: pose for pose in build_g1_visual_link_poses(stalled_segment)}

    assert grasp["right_shoulder_pitch_link"].rotate_xyz[1] > 15
    assert stalled["torso_link"].translate_xyz[2] < 0


def test_g1_link_path_candidates_include_nested_unitree_hierarchy():
    candidates = g1_link_path_candidates("/World/Robot/HumanoidAsset", "left_knee_link")

    assert "/World/Robot/HumanoidAsset/left_knee_link" in candidates
    assert "/World/Robot/HumanoidAsset/pelvis/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link" in candidates


def test_g1_link_path_candidates_include_torso_arm_hierarchy():
    candidates = g1_link_path_candidates("/World/Robot/HumanoidAsset", "right_elbow_link")

    assert "/World/Robot/HumanoidAsset/right_elbow_link" in candidates
    assert "/World/Robot/HumanoidAsset/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/right_elbow_link" in candidates
