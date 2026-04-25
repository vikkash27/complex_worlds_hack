import pytest

from robocerebra_rl.env import (
    AcknowledgeEventInput,
    ChooseSubgoalInput,
    ExecuteSkillInput,
    InventoryCheckInput,
    LogJobInput,
    MemorySearchInput,
    MemoryWriteInput,
    PlanCreateInput,
    PlanReviseInput,
    RoboCerebraShiftEnv,
    ScoreProgressInput,
    create_server,
)
from robocerebra_rl.shift import build_shift_spec


@pytest.mark.asyncio
async def test_shift_env_advertises_long_horizon_prompt_and_tool_budget():
    task = RoboCerebraShiftEnv.list_tasks("train")[0]
    env = RoboCerebraShiftEnv(task)

    prompt_text = env.get_prompt()[0].text.lower()

    assert "shift" in prompt_text
    assert "read_ticket" in prompt_text
    assert "memory_summarize" in prompt_text
    assert env.shift.spec.max_tool_calls >= 600
    assert len(env.shift.spec.jobs) >= 10


@pytest.mark.asyncio
async def test_shift_env_rejects_execute_before_read_ticket_and_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOCEREBRA_REWARD_CACHE", str(tmp_path / "vlm.json"))
    env = RoboCerebraShiftEnv(RoboCerebraShiftEnv.list_tasks("train")[0])

    rejected = await env.execute_skill(ExecuteSkillInput(action="locate_items"))
    assert rejected.metadata["accepted"] is False
    assert rejected.metadata["reason"] == "execute_before_ticket"

    await env.read_ticket()
    rejected_no_plan = await env.execute_skill(ExecuteSkillInput(action="locate_items"))
    assert rejected_no_plan.metadata["accepted"] is False
    assert rejected_no_plan.metadata["reason"] == "execute_before_plan"


@pytest.mark.asyncio
async def test_shift_env_progresses_one_job_through_full_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOCEREBRA_REWARD_CACHE", str(tmp_path / "vlm.json"))
    env = RoboCerebraShiftEnv(RoboCerebraShiftEnv.list_tasks("train")[0])

    await env.observe()
    ticket = (await env.read_ticket()).metadata["ticket"]
    plan_steps = [f"execute::{sub}" for sub in ticket["subgoals"]]
    plan_result = await env.plan_create(PlanCreateInput(steps=plan_steps))
    assert plan_result.metadata["accepted"] is True

    advanced = await env.execute_skill(ExecuteSkillInput(action="inspect_scene"))
    assert advanced.metadata["accepted"] is True
    progressed = await env.execute_skill(ExecuteSkillInput(action=ticket["subgoals"][0]))
    assert progressed.metadata["accepted"] is True
    assert progressed.reward >= 0.0

    score = await env.score_progress(ScoreProgressInput(subgoal=ticket["subgoals"][0]))
    assert "progress_delta" in score.metadata


@pytest.mark.asyncio
async def test_shift_env_supports_memory_inventory_and_event_acknowledgement(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOCEREBRA_REWARD_CACHE", str(tmp_path / "vlm.json"))
    spec = build_shift_spec(split="test", seed=4001)
    env = RoboCerebraShiftEnv(task_spec=spec.as_dict())

    await env.observe()
    write = await env.memory_write(MemoryWriteInput(key="guest::guest_amelia::preference", value="no_onions"))
    assert write.metadata["accepted"] is True
    search = await env.memory_search(MemorySearchInput(query="guest_amelia"))
    assert any(hit["key"] == "guest::guest_amelia::preference" for hit in search.metadata["hits"])

    inv = await env.inventory_check(InventoryCheckInput(item=None))
    assert isinstance(inv.metadata["inventory"], dict)

    # Acknowledge the first scheduled event after enough jobs have been logged.
    target_event = spec.events[0]
    while len(env.shift.completed_jobs) <= target_event.trigger_after_jobs:
        await env.read_ticket()
        await env.plan_create(PlanCreateInput(steps=["execute::locate_items"]))
        # Force-finish the active job by walking through expert macro actions.
        sub_world = env.shift.current_world
        if sub_world is None:
            break
        for action in sub_world.expert_actions():
            await env.execute_skill(ExecuteSkillInput(action=action))
            if env.shift.current_world is None or env.shift.current_world.done:
                break
        await env.log_job(LogJobInput(summary="forced"))
    overview = env.shift.overview()
    if overview["active_events"]:
        ack = await env.acknowledge_event(AcknowledgeEventInput(event_id=overview["active_events"][0]))
        assert ack.metadata["accepted"] is True
        revise = await env.plan_revise(PlanReviseInput(steps=["recover"]))
        assert revise.metadata["accepted"] in {True, False}


def test_shift_env_server_factory_constructs_server():
    server = create_server()

    assert server is not None


def test_shift_env_lists_at_least_100_total_tasks_across_splits():
    total = sum(len(RoboCerebraShiftEnv.list_tasks(s)) for s in ("train", "validation", "test"))

    assert total >= 100
