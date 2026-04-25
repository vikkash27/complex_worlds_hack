from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

from robocerebra_rl.rewards import symbolic_dense_reward
from robocerebra_rl.world import BreakfastTrayWorld, SceneConfig, iter_policy_actions


def randomized_world(seed: int) -> BreakfastTrayWorld:
    scene = SceneConfig.from_seed(seed)
    if seed >= 1000:
        scene = SceneConfig(
            mug_position=scene.mug_position,
            snack_position=scene.snack_position,
            tray_position=scene.tray_position,
            disturbance_tick=scene.disturbance_tick,
            distractor_count=max(2, scene.distractor_count),
            action_failure_prob=max(0.18, scene.action_failure_prob),
            disturbance_severity=max(0.7, scene.disturbance_severity),
        )
    return BreakfastTrayWorld(
        seed=seed,
        horizon_ticks=1000 + (seed % 3) * 250,
        max_macro_steps=18 if seed >= 1000 else 30,
        scene=scene,
    )


def ci95_for_rate(rate: float, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    half_width = 1.96 * math.sqrt(max(rate * (1.0 - rate), 0.0) / n)
    return [round(max(0.0, rate - half_width), 6), round(min(1.0, rate + half_width), 6)]


def evaluate_policy(
    policy: str | Iterable[str] | object,
    *,
    episodes: int = 20,
    seed: int = 0,
    randomized: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, float | int]:
    successes: list[float] = []
    progresses: list[float] = []
    rewards: list[float] = []
    ticks: list[int] = []
    recoveries: list[float] = []
    tool_calls: list[int] = []

    for index in range(episodes):
        world = randomized_world(seed + index) if randomized else BreakfastTrayWorld(seed=seed + index)
        total_reward = 0.0
        while not world.done:
            action = iter_policy_actions(policy, world)
            transition = world.step(action)
            total_reward += symbolic_dense_reward(transition)

        successes.append(1.0 if world.success else 0.0)
        progresses.append(world.progress_fraction)
        rewards.append(round(total_reward, 6))
        ticks.append(world.ticks)
        recoveries.append(1.0 if world.disturbance_recovered else 0.0)
        tool_calls.append(world.macro_steps)

    success_rate = round(mean(successes), 6)
    recovery_rate = round(mean(recoveries), 6)
    metrics: dict[str, float | int] = {
        "episodes": episodes,
        "task_regime": "randomized_heldout" if randomized else "deterministic_smoke_test",
        "success_rate": success_rate,
        "success_rate_ci95": ci95_for_rate(success_rate, episodes),
        "mean_progress": round(mean(progresses), 6),
        "mean_reward": round(mean(rewards), 6),
        "mean_ticks": round(mean(ticks), 6),
        "mean_tool_calls": round(mean(tool_calls), 6),
        "disturbance_recovery_rate": recovery_rate,
        "disturbance_recovery_ci95": ci95_for_rate(recovery_rate, episodes),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return metrics


def compare_policies(*, episodes: int = 60, seed: int = 0) -> dict[str, dict[str, float | int]]:
    return {
        "random": evaluate_policy("random", episodes=episodes, seed=seed, randomized=True),
        "fixed_script": evaluate_policy("fixed_script", episodes=episodes, seed=seed, randomized=True),
        "reactive_script": evaluate_policy("reactive_script", episodes=episodes, seed=seed, randomized=True),
        "expert_oracle": evaluate_policy("expert", episodes=episodes, seed=seed, randomized=True),
    }
