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
from robocerebra_rl.world import iter_policy_actions
from robocerebra_rl.world import BreakfastTrayWorld, SceneConfig


def evaluate_environment(
    *,
    environment_name: str,
    base_url: str | None,
    split: str,
    episodes: int,
    policy_name: str,
    output_path: Path,
) -> dict[str, object]:
    client = OpenReward(base_url=base_url) if base_url else OpenReward()
    environment = client.environments.get(name=environment_name)
    tasks = list(environment.list_tasks(split=split))
    if not tasks:
        raise RuntimeError(f"No tasks found for split {split!r} in {environment_name!r}")

    successes: list[float] = []
    rewards: list[float] = []
    tool_calls: list[int] = []
    rollout_rows: list[dict[str, object]] = []

    for episode in range(episodes):
        task = tasks[episode % len(tasks)]
        task_spec = getattr(task, "task_spec", task)
        local_world = _world_from_task_spec(task_spec)
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
                result = session.call_tool("execute_skill", {"action": action})
                calls += 1
                episode_reward += float(result.reward or 0.0)
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
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"metrics": metrics, "rollouts": rollout_rows}, indent=2), encoding="utf-8")
    return metrics


def _world_from_task_spec(task_spec: object) -> BreakfastTrayWorld:
    spec = task_spec if isinstance(task_spec, dict) else {}
    scene_spec = spec.get("scene") if isinstance(spec.get("scene"), dict) else {}
    scene = SceneConfig(
        mug_position=tuple(scene_spec.get("mug_position", SceneConfig().mug_position)),  # type: ignore[arg-type]
        snack_position=tuple(scene_spec.get("snack_position", SceneConfig().snack_position)),  # type: ignore[arg-type]
        tray_position=tuple(scene_spec.get("tray_position", SceneConfig().tray_position)),  # type: ignore[arg-type]
        disturbance_tick=int(scene_spec.get("disturbance_tick", SceneConfig().disturbance_tick)),
        distractor_count=int(scene_spec.get("distractor_count", SceneConfig().distractor_count)),
        action_failure_prob=float(scene_spec.get("action_failure_prob", SceneConfig().action_failure_prob)),
        disturbance_severity=float(scene_spec.get("disturbance_severity", SceneConfig().disturbance_severity)),
    )
    return BreakfastTrayWorld(
        seed=int(spec.get("seed", 0)),
        horizon_ticks=int(spec.get("horizon_ticks", 1000)),
        max_macro_steps=int(spec.get("max_macro_steps", 30)),
        scene=scene,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RoboCerebra Reward Lab through OpenReward.")
    parser.add_argument("--environment", default="vikkash/complex_worlds_hack")
    parser.add_argument("--base-url", default=None, help="Use http://127.0.0.1:8080 for local server.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", default="expert", choices=["expert", "reactive_script", "fixed_script", "random"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "openreward" / "benchmark_results.json")
    args = parser.parse_args()

    metrics = evaluate_environment(
        environment_name=args.environment,
        base_url=args.base_url,
        split=args.split,
        episodes=args.episodes,
        policy_name=args.policy,
        output_path=args.output,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
