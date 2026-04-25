from robocerebra_rl.humanoid_motion import (
    MotionSegment,
    compile_humanoid_motion,
    lane_status_counts,
    station_position,
    summarize_tool_trace,
)


def _execute_event(action: str, station: str, frame: int) -> dict[str, object]:
    return {
        "tool_name": "execute_skill",
        "action": action,
        "observation_summary": {
            "task_name": "humanoid_hospitality",
            "station": station,
            "phase": action.split("_", 1)[0],
            "frame_index": frame,
        },
    }


def _execute_event_with_reward(action: str, station: str, frame: int, reward: float, rationale: str = "") -> dict[str, object]:
    event = _execute_event(action, station, frame)
    event["reward"] = reward
    event["rationale"] = rationale
    return event


def test_station_position_maps_hospitality_workcells():
    assert station_position("pantry").x < station_position("counter").x
    assert station_position("delivery").x > station_position("table").x
    assert station_position("unknown").x == station_position("counter").x


def test_compile_humanoid_motion_builds_ordered_segments_with_gestures():
    events = [
        _execute_event("scan_pantry_1", "pantry", 12),
        _execute_event("navigate_counter_1", "counter", 24),
        _execute_event("grasp_counter_1", "counter", 36),
        _execute_event("place_table_1", "table", 48),
        _execute_event("verify_delivery_1", "delivery", 60),
    ]

    segments = compile_humanoid_motion(events)

    assert [segment.frame for segment in segments] == [12, 24, 36, 48, 60]
    assert [segment.gesture for segment in segments] == ["scan", "walk", "grasp", "place", "inspect"]
    assert all(isinstance(segment, MotionSegment) for segment in segments)
    assert segments[0].root_xyz[0] == station_position("pantry").x
    assert segments[-1].caption == "verify_delivery_1"


def test_compile_humanoid_motion_drops_non_execute_tool_events():
    events = [
        {"tool_name": "observe", "observation_summary": {"station": "pantry", "frame_index": 1}},
        _execute_event("report_delivery_1", "delivery", 72),
    ]

    assert len(compile_humanoid_motion(events)) == 1


def test_compile_humanoid_motion_marks_wait_failures_and_recovery_retries():
    events = [
        _execute_event_with_reward("wait", "counter", 84, -0.05, "Humanoid executes `wait`."),
        _execute_event_with_reward("scan_counter_1", "counter", 90, 0.08, "Recovery execution for `scan_counter_1`."),
        _execute_event_with_reward("navigate_counter_1", "counter", 96, 0.09),
    ]
    events[0]["observation_summary"]["expected_next"] = "scan_counter_1"

    segments = compile_humanoid_motion(events)

    assert [segment.status for segment in segments] == ["failed", "recovery", "success"]
    assert [segment.gesture for segment in segments] == ["stalled", "recover", "walk"]
    assert segments[1].frame > 90
    assert segments[2].frame > 96
    assert segments[0].caption == "WAIT: scan_counter_1"


def test_compile_humanoid_motion_preserves_optimized_no_wait_timeline():
    events = [
        _execute_event_with_reward("scan_counter_1", "counter", 84, 0.08),
        _execute_event_with_reward("navigate_counter_1", "counter", 96, 0.09),
    ]

    segments = compile_humanoid_motion(events)

    assert [segment.frame for segment in segments] == [84, 96]
    assert all(segment.status == "success" for segment in segments)


def test_summarize_tool_trace_counts_long_chain_and_stations():
    events = [
        {"tool_name": "observe", "observation_summary": {"station": "pantry"}},
        {"tool_name": "choose_subgoal", "observation_summary": {"station": "pantry"}},
        _execute_event("grasp_pantry_1", "pantry", 12),
        {"tool_name": "score_progress", "observation_summary": {"station": "pantry"}},
        _execute_event("place_table_1", "table", 24),
    ]

    summary = summarize_tool_trace(events)

    assert summary["total_tool_calls"] == 5
    assert summary["execute_skill_calls"] == 2
    assert summary["score_progress_calls"] == 1
    assert summary["stations"] == {"pantry": 4, "table": 1}


def test_summarize_tool_trace_counts_failures_and_recoveries():
    events = [
        _execute_event_with_reward("wait", "counter", 84, -0.05, "Humanoid waits."),
        _execute_event_with_reward("scan_counter_1", "counter", 90, 0.08, "Recovery execution for `scan_counter_1`."),
        _execute_event_with_reward("navigate_counter_1", "counter", 96, 0.09),
    ]

    summary = summarize_tool_trace(events)

    assert summary["failed_execute_calls"] == 1
    assert summary["recovery_execute_calls"] == 1


def test_lane_status_counts_summarizes_visible_route_markers():
    segments = compile_humanoid_motion(
        [
            _execute_event_with_reward("wait", "counter", 84, -0.05),
            _execute_event_with_reward("scan_counter_1", "counter", 90, 0.08, "Recovery execution."),
            _execute_event_with_reward("navigate_counter_1", "counter", 96, 0.09),
        ]
    )

    assert lane_status_counts(segments) == {"success": 1, "failed": 1, "recovery": 1}
