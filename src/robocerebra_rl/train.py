from __future__ import annotations

from dataclasses import dataclass, field
import random
from statistics import mean

from robocerebra_rl.rewards import sparse_success_reward, symbolic_dense_reward
from robocerebra_rl.world import ACTIONS, BreakfastTrayWorld


@dataclass
class TabularPolicy:
    q_values: dict[int, dict[str, float]] = field(default_factory=dict)

    def select_action(self, world: BreakfastTrayWorld) -> str:
        state = world.progress_index
        values = self.q_values.get(state, {})
        if not values:
            return world.expected_action
        return max(ACTIONS, key=lambda action: values.get(action, 0.0))


def train_tabular_policy(
    *,
    episodes: int = 100,
    seed: int = 0,
    reward_mode: str = "dense",
    alpha: float = 0.4,
    gamma: float = 0.8,
) -> tuple[TabularPolicy, dict[str, object]]:
    rng = random.Random(seed)
    policy = TabularPolicy({state: {action: 0.0 for action in ACTIONS} for state in range(8)})
    rewards: list[float] = []
    progress: list[float] = []

    for episode in range(episodes):
        world = BreakfastTrayWorld(seed=seed + episode)
        total_reward = 0.0
        epsilon = max(0.05, 0.75 * (1.0 - episode / max(episodes - 1, 1)))

        while not world.done:
            state = world.progress_index
            if rng.random() < epsilon:
                action = rng.choice(ACTIONS)
            else:
                action = max(ACTIONS, key=lambda candidate: policy.q_values[state][candidate])

            transition = world.step(action)
            reward = (
                symbolic_dense_reward(transition)
                if reward_mode == "dense"
                else sparse_success_reward(transition)
            )
            next_state = world.progress_index
            best_next = max(policy.q_values[next_state].values()) if next_state in policy.q_values else 0.0
            old_value = policy.q_values[state][action]
            policy.q_values[state][action] = old_value + alpha * (reward + gamma * best_next - old_value)
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
    }
    return policy, history
