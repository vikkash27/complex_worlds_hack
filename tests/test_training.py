from robocerebra_rl.eval import evaluate_policy
from robocerebra_rl.train import train_tabular_policy
from robocerebra_rl.world import BreakfastTrayWorld


def test_dense_training_improves_over_random_baseline():
    baseline = evaluate_policy("random", episodes=24, seed=2)
    learned_policy, history = train_tabular_policy(episodes=80, seed=2, reward_mode="dense")
    trained = evaluate_policy(learned_policy, episodes=24, seed=200)

    assert history["final_mean_reward"] > history["initial_mean_reward"]
    assert trained["mean_progress"] > baseline["mean_progress"]
    assert trained["success_rate"] >= baseline["success_rate"]


def test_episode_metrics_include_long_horizon_fields():
    policy = BreakfastTrayWorld(seed=1).expert_actions()

    metrics = evaluate_policy(policy, episodes=3, seed=5)

    assert metrics["episodes"] == 3
    assert metrics["mean_ticks"] >= 500
    assert "disturbance_recovery_rate" in metrics
