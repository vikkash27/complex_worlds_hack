from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openreward import OpenReward

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(override=False)

from robocerebra_rl.eval import ci95_for_rate
from robocerebra_rl.shift import (
    ShiftWorld,
    expert_shift_actions,
    random_shift_actions,
    reactive_shift_actions,
    shift_spec_from_dict,
)


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
        "events_handled_delta": round(
            float(improved_metrics.get("mean_events_handled", 0.0))
            - float(baseline_metrics.get("mean_events_handled", 0.0)),
            6,
        ),
        "memory_recalls_delta": round(
            float(improved_metrics.get("mean_memory_recalls", 0.0))
            - float(baseline_metrics.get("mean_memory_recalls", 0.0)),
            6,
        ),
        "tool_diversity_delta": round(
            float(improved_metrics.get("mean_tool_diversity", 0.0))
            - float(baseline_metrics.get("mean_tool_diversity", 0.0)),
            6,
        ),
        "baseline_total_tool_calls": int(baseline_metrics.get("total_tool_calls", 0)),
        "improved_total_tool_calls": int(improved_metrics.get("total_tool_calls", 0)),
        "aggregate_tool_calls": int(baseline_metrics.get("total_tool_calls", 0))
        + int(improved_metrics.get("total_tool_calls", 0)),
        "gemini_vision": gemini_metrics or {"enabled": False},
        "claim_boundary": (
            "Measures multi-job shift macro-policy tool use through OpenReward sessions. "
            "It is not low-level Isaac physics training."
        ),
    }


_POLICY_GENERATORS = {
    "expert": lambda world: expert_shift_actions(world),
    "reactive_script": lambda world: reactive_shift_actions(world),
    "random": lambda world: random_shift_actions(world, seed=world.spec.seed),
}


def _local_action_sequence(spec_dict: dict[str, Any], policy_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the policy on a local mirror to produce the action sequence and final stats."""
    local = ShiftWorld(spec=shift_spec_from_dict(spec_dict))
    if policy_name not in _POLICY_GENERATORS:
        raise ValueError(f"Unknown shift policy: {policy_name}")
    generator = _POLICY_GENERATORS[policy_name](local)
    sequence: list[dict[str, Any]] = []
    for call in generator:
        if local.done:
            break
        tool = call["tool"]
        params = call.get("params", {}) or {}
        method = getattr(local, tool)
        if isinstance(params, dict):
            method(**params)
        else:
            method(params)
        sequence.append({"tool": tool, "params": params})
        if local.done:
            break
    final = {
        "tool_calls": local.metrics.tool_calls,
        "events_handled": local.metrics.events_handled,
        "memory_recalls": local.metrics.memory_recalls,
        "plan_revisions": local.metrics.plan_revisions,
        "inventory_restocks": local.metrics.inventory_restocks,
        "score_progress_calls": local.metrics.score_progress_calls,
        "completed_jobs": len(local.completed_jobs),
        "failed_jobs": len(local.failed_jobs),
        "success": local.success,
        "tool_diversity": len(set(local.metrics.tool_call_log)),
    }
    return sequence, final


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
        tasks = [task for task in tasks if getattr(task, "task_spec", task).get("split") == task_name]
    if not tasks:
        raise RuntimeError(f"No tasks found for split {split!r} in {environment_name!r}")

    successes: list[float] = []
    rewards: list[float] = []
    tool_calls: list[int] = []
    events_handled: list[int] = []
    memory_recalls: list[int] = []
    score_progress_counts: list[int] = []
    plan_revisions: list[int] = []
    inventory_restocks: list[int] = []
    tool_diversity: list[int] = []
    rollout_rows: list[dict[str, object]] = []

    for episode in range(episodes):
        task = tasks[episode % len(tasks)]
        task_spec = getattr(task, "task_spec", task)
        spec_dict = task_spec if isinstance(task_spec, dict) else {}
        # Skip per-call score_progress for non-expert policies to keep budgets honest.
        sequence, local_final = _local_action_sequence(spec_dict, policy_name)
        if not score_progress:
            sequence = [c for c in sequence if c["tool"] != "score_progress"]
        episode_reward = 0.0

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

            calls = 0
            for call in sequence:
                tool = call["tool"]
                params = call.get("params") or {}
                result = session.call_tool(tool, params)
                calls += 1
                episode_reward += float(getattr(result, "reward", 0.0) or 0.0)
                rollout_rows.append(
                    {
                        "episode": episode,
                        "event": "tool_call",
                        "tool": tool,
                        "params": params,
                        "reward": getattr(result, "reward", None),
                        "finished": getattr(result, "finished", None),
                    }
                )
                if getattr(result, "finished", False):
                    break

        successes.append(1.0 if local_final["success"] else 0.0)
        rewards.append(round(episode_reward, 6))
        tool_calls.append(calls)
        events_handled.append(int(local_final["events_handled"]))
        memory_recalls.append(int(local_final["memory_recalls"]))
        score_progress_counts.append(int(local_final["score_progress_calls"]))
        plan_revisions.append(int(local_final["plan_revisions"]))
        inventory_restocks.append(int(local_final["inventory_restocks"]))
        tool_diversity.append(int(local_final["tool_diversity"]))

    success_rate = sum(successes) / len(successes)
    metrics = {
        "environment": environment_name,
        "base_url": base_url or "https://openreward.ai",
        "split": split,
        "episodes": episodes,
        "policy": policy_name,
        "success_rate": round(success_rate, 6),
        "success_rate_ci95": ci95_for_rate(success_rate, episodes),
        "mean_reward": round(mean(rewards), 6),
        "mean_tool_calls": round(mean(tool_calls), 6),
        "p50_tool_calls": round(sorted(tool_calls)[len(tool_calls) // 2], 6),
        "max_tool_calls": int(max(tool_calls)),
        "min_tool_calls": int(min(tool_calls)),
        "total_tool_calls": int(sum(tool_calls)),
        "mean_events_handled": round(mean(events_handled), 6),
        "mean_memory_recalls": round(mean(memory_recalls), 6),
        "mean_plan_revisions": round(mean(plan_revisions), 6),
        "mean_inventory_restocks": round(mean(inventory_restocks), 6),
        "mean_tool_diversity": round(mean(tool_diversity), 6),
        "mean_score_progress_calls": round(mean(score_progress_counts), 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"metrics": metrics, "rollouts": rollout_rows}, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RoboCerebra Reward Lab through OpenReward (shift mode).")
    parser.add_argument("--environment", default="vikkash/complex_worlds_hack")
    parser.add_argument("--base-url", default=None, help="Use http://127.0.0.1:8080 for local server.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--policy", default="expert", choices=["expert", "reactive_script", "random"])
    parser.add_argument("--compare-policy", default=None, choices=["expert", "reactive_script", "random"])
    parser.add_argument("--score-progress", action="store_true", help="Include `score_progress` calls in shift sequences.")
    parser.add_argument("--task-name", default=None, help="(Ignored in shift mode; kept for CLI compat.)")
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
            "metrics_primary": metrics,
            "metrics_compare": compare_metrics,
            "comparison": comparison,
        }
        submission_path.write_text(json.dumps(submission_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
