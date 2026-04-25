from __future__ import annotations

from dataclasses import dataclass, field
import random
from statistics import mean

from robocerebra_rl.rewards import sparse_success_reward, symbolic_dense_reward
from robocerebra_rl.eval import randomized_world
from robocerebra_rl.world import ACTIONS, BreakfastTrayWorld


@dataclass
class TabularPolicy:
    q_values: dict[object, dict[str, float]] = field(default_factory=dict)

    def select_action(self, world: BreakfastTrayWorld) -> str:
        state = policy_state(world)
        values = self.q_values.get(state, {})
        if not values:
            return world.expected_action
        return max(ACTIONS, key=lambda action: values.get(action, 0.0))


def policy_state(world: BreakfastTrayWorld) -> tuple[str, bool, bool, str]:
    return (
        world.expected_action,
        world.inspected,
        world.replanned,
        world.last_failure_reason or "",
    )


def ensure_state(policy: TabularPolicy, state: object) -> dict[str, float]:
    return policy.q_values.setdefault(state, {action: 0.0 for action in ACTIONS})


def train_tabular_policy(
    *,
    episodes: int = 100,
    seed: int = 0,
    reward_mode: str = "dense",
    randomized: bool = False,
    alpha: float = 0.4,
    gamma: float = 0.8,
) -> tuple[TabularPolicy, dict[str, object]]:
    rng = random.Random(seed)
    policy = TabularPolicy()
    rewards: list[float] = []
    progress: list[float] = []

    for episode in range(episodes):
        world = randomized_world(seed + episode) if randomized else BreakfastTrayWorld(seed=seed + episode)
        total_reward = 0.0
        epsilon = max(0.05, 0.75 * (1.0 - episode / max(episodes - 1, 1)))

        while not world.done:
            state = policy_state(world)
            values = ensure_state(policy, state)
            if rng.random() < epsilon:
                action = rng.choice(ACTIONS)
            else:
                action = max(ACTIONS, key=lambda candidate: values[candidate])

            transition = world.step(action)
            reward = (
                symbolic_dense_reward(transition)
                if reward_mode == "dense"
                else sparse_success_reward(transition)
            )
            next_state = policy_state(world)
            next_values = ensure_state(policy, next_state)
            best_next = max(next_values.values())
            old_value = values[action]
            values[action] = old_value + alpha * (reward + gamma * best_next - old_value)
            total_reward += reward

        rewards.append(round(total_reward, 6))
        progress.append(world.progress_fraction)

    window = min(10, len(rewards))
    history: dict[str, object] = {
        "episode_rewards": rewards,
        "episode_progress": progress,
        "initial_mean_reward": mean(rewards[:window]),
        "final_mean_reward": mean(rewards[-window:]),
        "best_reward": max(rewards),
        "episodes": episodes,
        "reward_mode": reward_mode,
        "task_regime": "randomized_heldout" if randomized else "deterministic_smoke_test",
    }
    return policy, history
