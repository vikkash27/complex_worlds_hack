from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from openreward import OpenReward

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(override=False)

from robocerebra_rl.eval import ci95_for_rate
from robocerebra_rl.world import iter_policy_actions, world_from_task_spec


def summarize_policy_comparison(
    *,
    baseline_name: str,
    improved_name: str,
    baseline_metrics: dict[str, object],
    improved_metrics: dict[str, object],
    gemini_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "baseline_policy": baseline_name,
        "improved_policy": improved_name,
        "success_lift": round(float(improved_metrics["success_rate"]) - float(baseline_metrics["success_rate"]), 6),
        "mean_reward_lift": round(float(improved_metrics["mean_reward"]) - float(baseline_metrics["mean_reward"]), 6),
        "tool_call_delta": round(float(improved_metrics["mean_tool_calls"]) - float(baseline_metrics["mean_tool_calls"]), 6),
        "baseline_total_tool_calls": int(baseline_metrics.get("total_tool_calls", 0)),
        "improved_total_tool_calls": int(improved_metrics.get("total_tool_calls", 0)),
        "aggregate_tool_calls": int(baseline_metrics.get("total_tool_calls", 0))
        + int(improved_metrics.get("total_tool_calls", 0)),
        "gemini_vision": gemini_metrics or {"enabled": False},
        "claim_boundary": (
            "Measures macro-policy tool use through OpenReward sessions. "
            "It is not low-level Isaac physics training."
        ),
    }


def evaluate_environment(
    *,
    environment_name: str,
    base_url: str | None,
    split: str,
    episodes: int,
    policy_name: str,
    output_path: Path,
    score_progress: bool = False,
    task_name: str | None = None,
) -> dict[str, object]:
    client = OpenReward(base_url=base_url) if base_url else OpenReward()
    environment = client.environments.get(name=environment_name)
    tasks = list(environment.list_tasks(split=split))
    if task_name:
        tasks = [task for task in tasks if getattr(task, "task_spec", task).get("task_name") == task_name]
    if not tasks:
        raise RuntimeError(f"No tasks found for split {split!r} in {environment_name!r}")

    successes: list[float] = []
    rewards: list[float] = []
    tool_calls: list[int] = []
    gemini_confidences: list[float] = []
    gemini_agreements: list[float] = []
    rollout_rows: list[dict[str, object]] = []

    for episode in range(episodes):
        task = tasks[episode % len(tasks)]
        task_spec = getattr(task, "task_spec", task)
        spec_dict = task_spec if isinstance(task_spec, dict) else {}
        local_world = world_from_task_spec(spec_dict)
        episode_reward = 0.0
        calls = 0

        with environment.session(task=task) as session:
            prompt = session.get_prompt()
            rollout_rows.append(
                {
                    "episode": episode,
                    "event": "prompt",
                    "prompt": getattr(prompt[0], "text", str(prompt[0])),
                    "task_spec": task_spec,
                }
            )

            while not local_world.done:
                action = iter_policy_actions(policy_name, local_world)
                local_transition = local_world.step(action)
                if score_progress:
                    observed = session.call_tool("observe", {})
                    calls += 1
                    rollout_rows.append({"episode": episode, "event": "observe", "finished": observed.finished})
                    chosen = session.call_tool("choose_subgoal", {"subgoal": local_transition.expected_action})
                    calls += 1
                    rollout_rows.append(
                        {
                            "episode": episode,
                            "event": "choose_subgoal",
                            "subgoal": local_transition.expected_action,
                            "finished": chosen.finished,
                        }
                    )
                result = session.call_tool("execute_skill", {"action": action})
                calls += 1
                episode_reward += float(result.reward or 0.0)
                if score_progress:
                    score = session.call_tool("score_progress", {"subgoal": local_transition.expected_action})
                    metadata = getattr(score, "metadata", {}) or {}
                    confidence = float(metadata.get("confidence", 0.0))
                    complete = bool(metadata.get("subgoal_complete", False))
                    expected_complete = local_transition.progress_delta > 0.0
                    gemini_confidences.append(confidence)
                    gemini_agreements.append(1.0 if complete == expected_complete else 0.0)
                    calls += 1
                    rollout_rows.append(
                        {
                            "episode": episode,
                            "event": "score_progress",
                            "subgoal": local_transition.expected_action,
                            "reward": score.reward,
                            "progress_delta": metadata.get("progress_delta"),
                            "subgoal_complete": metadata.get("subgoal_complete"),
                            "confidence": confidence,
                            "agreement": complete == expected_complete,
                            "rationale": metadata.get("rationale"),
                            "image_path": metadata.get("image_path"),
                        }
                    )
                rollout_rows.append(
                    {
                        "episode": episode,
                        "event": "tool_call",
                        "tool": "execute_skill",
                        "action": action,
                        "reward": result.reward,
                        "finished": result.finished,
                        "local_state_hash": local_transition.state_hash,
                        "blocks": [getattr(block, "text", str(block)) for block in result.blocks],
                    }
                )
                if result.finished:
                    break

        successes.append(1.0 if local_world.success else 0.0)
        rewards.append(round(episode_reward, 6))
        tool_calls.append(calls)

    success_rate = sum(successes) / len(successes)
    metrics = {
        "environment": environment_name,
        "base_url": base_url or "https://openreward.ai",
        "split": split,
        "episodes": episodes,
        "policy": policy_name,
        "success_rate": round(success_rate, 6),
        "success_rate_ci95": ci95_for_rate(success_rate, episodes),
        "mean_reward": round(sum(rewards) / len(rewards), 6),
        "mean_tool_calls": round(sum(tool_calls) / len(tool_calls), 6),
        "total_tool_calls": int(sum(tool_calls)),
    }
    if score_progress and gemini_confidences:
        metrics["gemini_vision"] = {
            "enabled": True,
            "mean_confidence": round(sum(gemini_confidences) / len(gemini_confidences), 6),
            "agreement_rate": round(sum(gemini_agreements) / len(gemini_agreements), 6),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"metrics": metrics, "rollouts": rollout_rows}, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RoboCerebra Reward Lab through OpenReward.")
    parser.add_argument("--environment", default="vikkash/complex_worlds_hack")
    parser.add_argument("--base-url", default=None, help="Use http://127.0.0.1:8080 for local server.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", default="expert", choices=["expert", "reactive_script", "fixed_script", "random"])
    parser.add_argument("--compare-policy", default=None, choices=["expert", "reactive_script", "fixed_script", "random"])
    parser.add_argument("--score-progress", action="store_true", help="Call score_progress after each execute_skill.")
    parser.add_argument("--task-name", default=None, help="Filter OpenReward tasks by task_name, e.g. humanoid_hospitality.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "openreward" / "benchmark_results.json")
    args = parser.parse_args()

    metrics = evaluate_environment(
        environment_name=args.environment,
        base_url=args.base_url,
        split=args.split,
        episodes=args.episodes,
        policy_name=args.policy,
        output_path=args.output,
        score_progress=args.score_progress,
        task_name=args.task_name,
    )
    output: dict[str, object] = {"metrics": metrics}
    if args.compare_policy:
        compare_output = args.output.with_name(f"{args.compare_policy}_comparison_results.json")
        compare_metrics = evaluate_environment(
            environment_name=args.environment,
            base_url=args.base_url,
            split=args.split,
            episodes=args.episodes,
            policy_name=args.compare_policy,
            output_path=compare_output,
            score_progress=args.score_progress,
            task_name=args.task_name,
        )
        comparison = summarize_policy_comparison(
            baseline_name=args.policy,
            improved_name=args.compare_policy,
            baseline_metrics=metrics,
            improved_metrics=compare_metrics,
            gemini_metrics=compare_metrics.get("gemini_vision") if isinstance(compare_metrics.get("gemini_vision"), dict) else None,
        )
        output["comparison"] = comparison
        summary_path = args.output.with_name(f"{args.output.stem}_comparison_summary.json")
        summary_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        submission_path = args.output.parent / "submission_benchmark_summary.json"
        submission_payload = {
            "environment": args.environment,
            "split": args.split,
            "episodes": args.episodes,
            "score_progress": args.score_progress,
            "task_name_filter": args.task_name,
            "metrics_primary": metrics,
            "metrics_compare": compare_metrics,
            "comparison": comparison,
        }
        submission_path.write_text(json.dumps(submission_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
