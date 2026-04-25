import pytest

from robocerebra_rl.env import ExecuteSkillInput, RoboCerebraRewardLabEnv, create_server


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


def test_openreward_server_factory_constructs_server():
    server = create_server()

    assert server is not None
