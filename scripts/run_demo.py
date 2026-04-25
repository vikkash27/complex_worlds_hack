from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(override=False)

from robocerebra_rl.eval import evaluate_policy
from robocerebra_rl.eval import compare_policies, randomized_world
from robocerebra_rl.render import render_world, save_replay
from robocerebra_rl.rewards import GeminiRewardCache, gemini_reward_scorer, symbolic_dense_reward
from robocerebra_rl.trace import ToolTraceLogger
from robocerebra_rl.train import train_tabular_policy
from robocerebra_rl.world import BreakfastTrayWorld, iter_policy_actions


ARTIFACTS = ROOT / "artifacts"


def rollout_frames(
    policy: str | object,
    seed: int,
    output_frame_dir: Path,
    trace_path: Path,
    *,
    randomized: bool = True,
) -> tuple[list, dict[str, float]]:
    world = randomized_world(seed) if randomized else BreakfastTrayWorld(seed=seed, horizon_ticks=1000)
    trace = ToolTraceLogger(trace_path, run_id=f"replay-{seed}")
    frames = [render_world(world)]
    total_reward = 0.0
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    render_world(world, output_frame_dir / "frame_000.png")
    trace.record(
        tool_name="observe",
        task_id=world.task.task_id,
        action=None,
        observation=world.observe(),
        reward=0.0,
        reward_components={},
        rationale="Initial scene observation before any macro-action.",
        finished=False,
        state_hash=world.state_hash(),
    )

    while not world.done:
        action = iter_policy_actions(policy, world)
        transition = world.step(action)
        reward = symbolic_dense_reward(transition)
        total_reward += reward
        trace.record(
            tool_name="execute_skill",
            task_id=world.task.task_id,
            action=action,
            observation=transition.observation,
            reward=reward,
            reward_components={
                "progress": round(transition.progress_delta * 1.5, 6),
                "success": 1.0 if transition.success else 0.0,
            },
            rationale=(
                f"{action} advanced progress by {transition.progress_delta:.3f}."
                if transition.progress_delta > 0
                else f"{action} failed or was not applicable: {transition.observation.get('last_failure_reason') or 'not expected now'}."
            ),
            finished=transition.done,
            state_hash=transition.state_hash,
        )
        frame = render_world(world)
        frames.append(frame)
        render_world(world, output_frame_dir / f"frame_{world.macro_steps:03d}.png")

    return frames, {
        "success": 1.0 if world.success else 0.0,
        "progress": world.progress_fraction,
        "ticks": float(world.ticks),
        "reward": round(total_reward, 6),
        "disturbance_recovered": 1.0 if world.disturbance_recovered else 0.0,
        "tool_calls": float(world.macro_steps),
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
    for stale_dir in [
        ARTIFACTS / "metrics",
        ARTIFACTS / "plots",
        ARTIFACTS / "replays",
        ARTIFACTS / "traces",
        ARTIFACTS / "visual_report",
    ]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)

    metrics_dir = ARTIFACTS / "metrics"
    plots_dir = ARTIFACTS / "plots"
    replays_dir = ARTIFACTS / "replays"
    cache_dir = ARTIFACTS / "cache"

    smoke_policy, smoke_history = train_tabular_policy(episodes=120, seed=13, reward_mode="dense")
    dense_policy, dense_history = train_tabular_policy(
        episodes=200,
        seed=13,
        reward_mode="dense",
        randomized=True,
    )
    sparse_policy, sparse_history = train_tabular_policy(
        episodes=200,
        seed=13,
        reward_mode="sparse",
        randomized=True,
    )
    smoke_random_metrics = evaluate_policy("random", episodes=60, seed=100)
    smoke_trained_metrics = evaluate_policy(smoke_policy, episodes=60, seed=200)
    randomized_baselines = compare_policies(episodes=80, seed=1000)
    dense_trained_metrics = evaluate_policy(dense_policy, episodes=80, seed=2000, randomized=True)
    sparse_trained_metrics = evaluate_policy(sparse_policy, episodes=80, seed=2000, randomized=True)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "smoke_random_baseline.json").write_text(
        json.dumps(smoke_random_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metrics_dir / "smoke_dense_trained_policy.json").write_text(
        json.dumps(smoke_trained_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metrics_dir / "randomized_policy_report.json").write_text(
        json.dumps(
            {
                **randomized_baselines,
                "dense_trained": dense_trained_metrics,
                "sparse_trained": sparse_trained_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (metrics_dir / "training_history.json").write_text(
        json.dumps(
            {
                "deterministic_smoke_test": smoke_history,
                "randomized_dense": dense_history,
                "randomized_sparse": sparse_history,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sample_vlm_score = score_sample_transition(cache_dir / "gemini_reward_cache.json")
    leaderboard = {
        "benchmark": "RoboCerebra Reward Lab",
        "task": "breakfast_tray_disturbance",
        "result_type": "randomized_heldout_benchmark",
        "deterministic_smoke_test": {
            "random_baseline": smoke_random_metrics,
            "dense_trained_policy": smoke_trained_metrics,
        },
        "randomized_heldout": {
            **randomized_baselines,
            "dense_trained": dense_trained_metrics,
            "sparse_trained": sparse_trained_metrics,
        },
        "sample_vlm_reward": sample_vlm_score,
        "headline": {
            "progress_lift": round(
                float(dense_trained_metrics["mean_progress"]) - float(randomized_baselines["reactive_script"]["mean_progress"]),
                6,
            ),
            "success_lift": round(
                float(dense_trained_metrics["success_rate"]) - float(randomized_baselines["reactive_script"]["success_rate"]),
                6,
            ),
            "reward_auc_lift": round(
                float(dense_history["final_mean_reward"]) - float(dense_history["initial_mean_reward"]),
                6,
            ),
        },
    }
    (metrics_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_plot(dense_history, float(randomized_baselines["reactive_script"]["mean_reward"]), plots_dir / "training_curve.png")

    traces_dir = ARTIFACTS / "traces"
    baseline_frames, baseline_rollout = rollout_frames(
        "fixed_script",
        501,
        replays_dir / "baseline_frames",
        traces_dir / "baseline_fixed_script.jsonl",
    )
    trained_frames, trained_rollout = rollout_frames(
        dense_policy,
        501,
        replays_dir / "trained_frames",
        traces_dir / "dense_trained.jsonl",
    )
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
    print("Trace artifacts:")
    print(f"- {traces_dir / 'baseline_fixed_script.jsonl'}")
    print(f"- {traces_dir / 'dense_trained.jsonl'}")


if __name__ == "__main__":
    main()
