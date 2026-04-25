import json

from robocerebra_rl.trace import ToolTraceLogger
from robocerebra_rl.world import BreakfastTrayWorld


def test_trace_logger_records_tool_call_reward_and_rationale(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    logger = ToolTraceLogger(trace_path, run_id="unit-test")
    world = BreakfastTrayWorld(seed=5)
    transition = world.step("locate_items")

    logger.record(
        tool_name="execute_skill",
        task_id=world.task.task_id,
        action="locate_items",
        observation=transition.observation,
        reward=0.25,
        reward_components={"progress": 0.2, "success": 0.0},
        rationale="locate_items completed the first subgoal",
        finished=False,
        state_hash=transition.state_hash,
    )

    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["tool_name"] == "execute_skill"
    assert event["reward_components"]["progress"] == 0.2
    assert event["rationale"].startswith("locate_items")
    assert event["state_hash"] == transition.state_hash
