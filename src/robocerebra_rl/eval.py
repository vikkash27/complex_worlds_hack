from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from robocerebra_rl.rewards import symbolic_dense_reward
from robocerebra_rl.world import BreakfastTrayWorld, iter_policy_actions


def evaluate_policy(
    policy: str | Iterable[str] | object,
    *,
    episodes: int = 20,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> dict[str, float | int]:
    successes: list[float] = []
    progresses: list[float] = []
    rewards: list[float] = []
    ticks: list[int] = []
    recoveries: list[float] = []

    for index in range(episodes):
        world = BreakfastTrayWorld(seed=seed + index)
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

    metrics: dict[str, float | int] = {
        "episodes": episodes,
        "success_rate": round(mean(successes), 6),
        "mean_progress": round(mean(progresses), 6),
        "mean_reward": round(mean(rewards), 6),
        "mean_ticks": round(mean(ticks), 6),
        "disturbance_recovery_rate": round(mean(recoveries), 6),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return metrics
