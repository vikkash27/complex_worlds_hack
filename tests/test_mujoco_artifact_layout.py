from robocerebra_rl.humanoid_motion import MotionSegment
from robocerebra_rl.mujoco_artifact_layout import artifact_positions_for_segment, default_artifact_home_positions


def _seg(**kwargs: object) -> MotionSegment:
    return MotionSegment(
        frame=10,
        action=str(kwargs.get("action", "navigate_counter_0")),
        station=str(kwargs.get("station", "counter")),
        phase=str(kwargs.get("phase", "navigate")),
        gesture=str(kwargs.get("gesture", "walk")),
        root_xyz=(0.0, 0.0, 0.0),
        left_hand_xyz=(0.0, 0.0, 0.0),
        right_hand_xyz=(0.0, 0.0, 0.0),
        caption="",
        status=str(kwargs.get("status", "success")),
        progress_fraction=0.2,
        tool_call_index=1,
    )


def test_artifact_positions_have_mocap_targets() -> None:
    pos = artifact_positions_for_segment(_seg(), frame_index=0)
    for key in ("spill", "tote"):
        assert key in pos
        assert len(pos[key]) == 3
    assert "tray" in pos


def test_spill_hidden_when_not_at_sink() -> None:
    p = artifact_positions_for_segment(_seg(station="counter", phase="place"), frame_index=0)
    # Spill stacked under floor
    assert p["spill"][1] < -1.0


def test_tray_on_delivery_station() -> None:
    p = artifact_positions_for_segment(_seg(station="delivery", gesture="handoff"), frame_index=0)
    # Delivery bench x ~ 0.52
    assert p["tray"][0] > 0.3


def test_default_home_has_four_props() -> None:
    h = default_artifact_home_positions()
    assert len(h) == 4
