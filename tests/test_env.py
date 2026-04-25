import pytest

from robocerebra_rl.env import ExecuteSkillInput, RoboCerebraRewardLabEnv, ScoreProgressInput, create_server


@pytest.mark.asyncio
async def test_openreward_environment_exposes_tools_and_rewards():
    task = RoboCerebraRewardLabEnv.list_tasks("train")[0]
    env = RoboCerebraRewardLabEnv(task)

    prompt = env.get_prompt()
    observation = await env.observe()
    await env.execute_skill(ExecuteSkillInput(action="inspect_scene"))
    result = await env.execute_skill(ExecuteSkillInput(action="locate_items"))

    assert "breakfast tray" in prompt[0].text.lower()
    assert observation.reward == 0.0
    assert result.reward > 0.0
    assert result.finished is False
    assert "locate_items" in result.metadata["completed_subgoals"]


@pytest.mark.asyncio
async def test_score_progress_records_rendered_image_for_gemini_vision(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOCEREBRA_REWARD_CACHE", str(tmp_path / "reward_cache.json"))
    env = RoboCerebraRewardLabEnv(RoboCerebraRewardLabEnv.list_tasks("train")[0])

    await env.execute_skill(ExecuteSkillInput(action="inspect_scene"))
    await env.execute_skill(ExecuteSkillInput(action="locate_items"))
    result = await env.score_progress(ScoreProgressInput(subgoal="locate_items"))

    image_path = result.metadata["image_path"]
    assert image_path.endswith(".png")
    assert (tmp_path / image_path.rsplit("/", 1)[-1]).exists()
    assert result.metadata["progress_delta"] > 0
    assert result.metadata["subgoal_complete"] is True
    assert result.metadata["confidence"] > 0


def test_openreward_server_factory_constructs_server():
    server = create_server()

    assert server is not None


def test_openreward_task_list_includes_showcase_task_names():
    tasks = RoboCerebraRewardLabEnv.list_tasks("test")
    task_names = {task["task_name"] for task in tasks}

    assert {"breakfast_tray", "spill_recovery", "countertop_cleanup", "humanoid_hospitality"}.issubset(task_names)
    assert all("task_name" in task for task in tasks)


def test_prompt_uses_selected_task_instruction():
    task = next(task for task in RoboCerebraRewardLabEnv.list_tasks("test") if task["task_name"] == "countertop_cleanup")
    env = RoboCerebraRewardLabEnv(task)

    prompt_text = env.get_prompt()[0].text.lower()
    assert "countertop cleanup" in prompt_text
    assert "breakfast tray task" not in prompt_text
