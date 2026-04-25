from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.eval import evaluate_policy
from robocerebra_rl.render import render_world, save_replay
from robocerebra_rl.rewards import GeminiRewardCache, gemini_reward_scorer, symbolic_dense_reward
from robocerebra_rl.train import train_tabular_policy
from robocerebra_rl.world import BreakfastTrayWorld, iter_policy_actions


ARTIFACTS = ROOT / "artifacts"


def rollout_frames(policy: str | object, seed: int, output_frame_dir: Path) -> tuple[list, dict[str, float]]:
    world = BreakfastTrayWorld(seed=seed, horizon_ticks=1000)
    frames = [render_world(world)]
    total_reward = 0.0
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    render_world(world, output_frame_dir / "frame_000.png")

    while not world.done:
        action = iter_policy_actions(policy, world)
        transition = world.step(action)
        total_reward += symbolic_dense_reward(transition)
        frame = render_world(world)
        frames.append(frame)
        render_world(world, output_frame_dir / f"frame_{world.macro_steps:03d}.png")

    return frames, {
        "success": 1.0 if world.success else 0.0,
        "progress": world.progress_fraction,
        "ticks": float(world.ticks),
        "reward": round(total_reward, 6),
        "disturbance_recovered": 1.0 if world.disturbance_recovered else 0.0,
    }


def write_plot(history: dict[str, object], baseline_reward: float, path: Path) -> None:
    rewards = list(history["episode_rewards"])
    window = 10
    moving = [
        sum(rewards[max(0, i - window + 1) : i + 1]) / len(rewards[max(0, i - window + 1) : i + 1])
        for i in range(len(rewards))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    plt.plot(rewards, alpha=0.25, label="Dense reward per episode")
    plt.plot(moving, linewidth=2.5, label="10-episode moving average")
    plt.axhline(baseline_reward, linestyle="--", color="black", label="Random baseline mean")
    plt.title("Dense Gemini-style Reward Speeds Macro-Policy Learning")
    plt.xlabel("Training episode")
    plt.ylabel("Dense reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def score_sample_transition(path: Path) -> dict[str, object]:
    world = BreakfastTrayWorld(seed=42)
    transition = world.step("locate_items")
    cache = GeminiRewardCache(path, scorer=gemini_reward_scorer())
    return cache.score(
        world.task.task_id,
        transition.state_hash,
        "locate_items",
        "locate_items",
        progress_delta=transition.progress_delta,
    )


def main() -> None:
    metrics_dir = ARTIFACTS / "metrics"
    plots_dir = ARTIFACTS / "plots"
    replays_dir = ARTIFACTS / "replays"
    cache_dir = ARTIFACTS / "cache"

    policy, history = train_tabular_policy(episodes=120, seed=13, reward_mode="dense")
    random_metrics = evaluate_policy("random", episodes=60, seed=100)
    trained_metrics = evaluate_policy(policy, episodes=60, seed=200)
    expert_metrics = evaluate_policy("expert", episodes=10, seed=300)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "random_baseline.json").write_text(
        json.dumps(random_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metrics_dir / "dense_trained_policy.json").write_text(
        json.dumps(trained_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metrics_dir / "expert_oracle.json").write_text(
        json.dumps(expert_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metrics_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    sample_vlm_score = score_sample_transition(cache_dir / "gemini_reward_cache.json")
    leaderboard = {
        "benchmark": "RoboCerebra Reward Lab",
        "task": "breakfast_tray_disturbance",
        "horizon_ticks": 1000,
        "random_baseline": random_metrics,
        "dense_trained_policy": trained_metrics,
        "expert_oracle": expert_metrics,
        "sample_vlm_reward": sample_vlm_score,
        "headline": {
            "progress_lift": round(
                float(trained_metrics["mean_progress"]) - float(random_metrics["mean_progress"]),
                6,
            ),
            "success_lift": round(
                float(trained_metrics["success_rate"]) - float(random_metrics["success_rate"]),
                6,
            ),
            "reward_auc_lift": round(
                float(history["final_mean_reward"]) - float(history["initial_mean_reward"]),
                6,
            ),
        },
    }
    (metrics_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_plot(history, float(random_metrics["mean_reward"]), plots_dir / "training_curve.png")

    baseline_frames, baseline_rollout = rollout_frames("random", 501, replays_dir / "baseline_frames")
    trained_frames, trained_rollout = rollout_frames(policy, 501, replays_dir / "trained_frames")
    save_replay(baseline_frames, replays_dir / "baseline_random.gif")
    save_replay(trained_frames, replays_dir / "dense_trained.gif")
    (metrics_dir / "replay_rollouts.json").write_text(
        json.dumps(
            {"baseline_random": baseline_rollout, "dense_trained": trained_rollout},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("RoboCerebra Reward Lab demo artifacts written:")
    print(f"- {metrics_dir / 'leaderboard.json'}")
    print(f"- {plots_dir / 'training_curve.png'}")
    print(f"- {replays_dir / 'baseline_random.gif'}")
    print(f"- {replays_dir / 'dense_trained.gif'}")
    print(json.dumps(leaderboard["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
