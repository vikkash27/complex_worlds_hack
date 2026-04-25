from robocerebra_rl.eval import compare_policies, evaluate_policy
from robocerebra_rl.world import BreakfastTrayWorld, SceneConfig


def test_randomized_scene_requires_observation_before_locating_distracted_items():
    world = BreakfastTrayWorld(
        seed=12,
        scene=SceneConfig(distractor_count=2, action_failure_prob=0.0),
    )

    blocked = world.step("locate_items")
    observed = world.step("inspect_scene")
    progressed = world.step("locate_items")

    assert blocked.progress_delta == 0.0
    assert blocked.observation["last_failure_reason"] == "needs_inspection"
    assert observed.observation["inspected"] is True
    assert progressed.progress_delta > 0.0


def test_randomized_eval_reports_stronger_baselines_and_confidence_intervals():
    report = compare_policies(episodes=24, seed=31)

    assert "fixed_script" in report
    assert "reactive_script" in report
    assert "expert_oracle" in report
    assert report["random"]["success_rate"] < report["expert_oracle"]["success_rate"]
    assert report["fixed_script"]["success_rate"] < report["expert_oracle"]["success_rate"]
    assert report["reactive_script"]["success_rate"] >= report["fixed_script"]["success_rate"]
    assert "success_rate_ci95" in report["expert_oracle"]
    assert report["expert_oracle"]["mean_tool_calls"] >= 7


def test_policy_evaluation_can_run_on_randomized_heldout_tasks():
    metrics = evaluate_policy("expert", episodes=10, seed=1000, randomized=True)

    assert metrics["episodes"] == 10
    assert metrics["task_regime"] == "randomized_heldout"
    assert metrics["success_rate"] >= 0.6
    assert metrics["mean_tool_calls"] >= 7
