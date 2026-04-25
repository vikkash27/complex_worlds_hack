import json

from robocerebra_rl.isaac_scene import (
    humanoid_asset_candidates,
    make_humanoid_showcase_scene_plan,
)
from robocerebra_rl.world import ACTIONS, TASK_LIBRARY, BreakfastTrayWorld
from scripts.isaac.replay_breakfast_tray import asset_candidates_custom_data, load_trace_events, should_use_humanoid_showcase
from scripts.run_demo import generate_humanoid_showcase_trace


def test_humanoid_assets_prioritize_unitree_g1(monkeypatch):
    monkeypatch.delenv("ROBOCEREBRA_HUMANOID_USD", raising=False)
    candidates = humanoid_asset_candidates()
    assert candidates[0] == "/Isaac/Robots/Unitree/G1/g1.usd"
    assert any("Unitree/H1" in c for c in candidates)
    assert any("Humanoid" in c for c in candidates)

    plan = make_humanoid_showcase_scene_plan(baseline_events=[], trained_events=[])
    assert plan.title == "RoboCerebra Humanoid OpenReward Showcase"
    assert plan.humanoid_asset_candidates == candidates
    assert plan.camera.position[2] > 3.0


def test_robocerebra_humanoid_usd_override_prepends(monkeypatch):
    monkeypatch.setenv("ROBOCEREBRA_HUMANOID_USD", "/workspace/artifacts/isaac/robots/g1.usd")
    candidates = humanoid_asset_candidates()
    assert candidates[0] == "/workspace/artifacts/isaac/robots/g1.usd"
    assert "/Isaac/Robots/Unitree/G1/g1.usd" in candidates


def test_humanoid_hospitality_task_has_long_horizon_micro_actions():
    template = TASK_LIBRARY["humanoid_hospitality"]

    assert len(template.subgoals) >= 30
    assert all(action in ACTIONS for action in template.subgoals)

    world = BreakfastTrayWorld(seed=77, task_name="humanoid_hospitality", max_macro_steps=200)
    assert "hospitality" in world.observe()["instruction"].lower()
    assert world.expert_actions() == list(template.subgoals)


def test_generate_humanoid_showcase_trace_has_100_plus_events(tmp_path):
    output = tmp_path / "humanoid_trained.jsonl"

    metrics = generate_humanoid_showcase_trace(output, optimized=True)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert metrics["tool_events"] >= 100
    assert metrics["execute_skill_calls"] >= 30
    assert any(row["tool_name"] == "choose_subgoal" for row in rows)
    assert any(row["tool_name"] == "score_progress" for row in rows)
    assert all("station" in row["observation_summary"] for row in rows if row["tool_name"] == "execute_skill")


def test_humanoid_trace_is_detected_by_isaac_replay(tmp_path):
    output = tmp_path / "humanoid_trained.jsonl"
    generate_humanoid_showcase_trace(output, optimized=True)

    events = load_trace_events(output)

    assert should_use_humanoid_showcase(events) is True
    assert len(events) >= 100


def test_asset_candidates_custom_data_is_usd_safe_json_string():
    candidates = ("/Isaac/Robots/Unitree/H1/h1.usd", "/Isaac/Robots/IsaacSim/Humanoid/humanoid.usd")

    custom_data = asset_candidates_custom_data(candidates)

    assert isinstance(custom_data, str)
    assert json.loads(custom_data) == list(candidates)
