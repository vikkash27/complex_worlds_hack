from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Iterable


SUBGOALS = [
    "locate_items",
    "clear_workspace",
    "pick_mug",
    "fill_drink",
    "place_snack",
    "recover_disturbance",
    "deliver_tray",
]

ACTIONS = [
    *SUBGOALS,
    "inspect_scene",
    "wait",
    "replan",
]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    subgoals: tuple[str, ...] = tuple(SUBGOALS)
    horizon_ticks: int = 1000


@dataclass(frozen=True)
class Transition:
    action: str
    expected_action: str
    ticks: int
    progress_before: float
    progress_after: float
    progress_delta: float
    reward: float
    done: bool
    success: bool
    disturbance_recovered: bool
    state_hash: str
    observation: dict[str, object]


@dataclass
class BreakfastTrayWorld:
    seed: int = 0
    horizon_ticks: int = 1000
    max_macro_steps: int = 18
    task: TaskSpec = field(init=False)
    ticks: int = field(init=False, default=0)
    macro_steps: int = field(init=False, default=0)
    progress_index: int = field(init=False, default=0)
    success: bool = field(init=False, default=False)
    done: bool = field(init=False, default=False)
    disturbance_recovered: bool = field(init=False, default=False)
    action_history: list[str] = field(init=False, default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.task = TaskSpec(
            task_id=f"breakfast-tray-{self.seed}",
            instruction=(
                "Prepare and deliver a breakfast tray. Locate the mug and snack, "
                "clear the workspace, fill the drink, recover from the tray bump, "
                "and deliver without spilling."
            ),
            horizon_ticks=self.horizon_ticks,
        )

    @property
    def progress_fraction(self) -> float:
        return round(self.progress_index / len(self.task.subgoals), 6)

    @property
    def expected_action(self) -> str:
        if self.progress_index >= len(self.task.subgoals):
            return "submit_done"
        return self.task.subgoals[self.progress_index]

    @property
    def macro_tick_size(self) -> int:
        return max(25, math.ceil(self.horizon_ticks / len(self.task.subgoals)))

    def expert_actions(self) -> list[str]:
        return list(self.task.subgoals)

    def state_hash(self) -> str:
        raw = "|".join(
            [
                self.task.task_id,
                str(self.progress_index),
                str(self.ticks),
                str(self.disturbance_recovered),
                ",".join(self.action_history[-5:]),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def observe(self) -> dict[str, object]:
        completed = list(self.task.subgoals[: self.progress_index])
        remaining = list(self.task.subgoals[self.progress_index :])
        return {
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "ticks": self.ticks,
            "macro_steps": self.macro_steps,
            "horizon_ticks": self.horizon_ticks,
            "completed_subgoals": completed,
            "remaining_subgoals": remaining,
            "expected_next": self.expected_action,
            "progress_fraction": self.progress_fraction,
            "disturbance_recovered": self.disturbance_recovered,
            "success": self.success,
            "done": self.done,
        }

    def step(self, action: str) -> Transition:
        if self.done:
            return self._transition(action, self.expected_action, 0.0, -0.1)

        expected = self.expected_action
        before = self.progress_fraction
        self.action_history.append(action)
        self.macro_steps += 1
        self.ticks += self.macro_tick_size

        if action == expected:
            self.progress_index += 1
            if action == "recover_disturbance":
                self.disturbance_recovered = True
            progress_delta = self.progress_fraction - before
            reward = 0.25 + progress_delta + (0.1 if self.disturbance_recovered else 0.0)
        elif action in {"inspect_scene", "replan"}:
            progress_delta = 0.0
            reward = -0.01
        else:
            progress_delta = 0.0
            reward = -0.08

        if self.progress_index >= len(self.task.subgoals):
            self.success = self.disturbance_recovered
            self.done = True
            self.ticks = max(self.ticks, self.horizon_ticks)
            reward += 1.0 if self.success else -0.5
        elif self.macro_steps >= self.max_macro_steps:
            self.done = True
            reward -= 0.25

        return self._transition(action, expected, progress_delta, reward)

    def _transition(
        self, action: str, expected_action: str, progress_delta: float, reward: float
    ) -> Transition:
        after = self.progress_fraction
        return Transition(
            action=action,
            expected_action=expected_action,
            ticks=self.ticks,
            progress_before=round(after - progress_delta, 6),
            progress_after=after,
            progress_delta=round(progress_delta, 6),
            reward=round(reward, 6),
            done=self.done,
            success=self.success,
            disturbance_recovered=self.disturbance_recovered,
            state_hash=self.state_hash(),
            observation=self.observe(),
        )


def iter_policy_actions(policy: str | Iterable[str] | object, world: BreakfastTrayWorld) -> str:
    if isinstance(policy, str):
        if policy == "expert":
            return world.expected_action
        if policy == "random":
            return world._rng.choice(ACTIONS)
        if policy in ACTIONS:
            return policy
        raise ValueError(f"Unknown policy string: {policy}")
    if hasattr(policy, "select_action"):
        return policy.select_action(world)
    if not hasattr(policy, "__iter__"):
        raise TypeError(f"Unsupported policy type: {type(policy)!r}")
    actions = list(policy)
    if not actions:
        return "wait"
    index = min(world.macro_steps, len(actions) - 1)
    return actions[index]
