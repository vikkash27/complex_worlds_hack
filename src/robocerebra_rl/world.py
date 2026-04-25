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

SHOWCASE_SUBGOALS = [
    "stabilize_spill",
    "place_absorbent_pad",
    "scan_countertop",
    "sort_recyclables",
    "place_utensils",
    "wipe_countertop",
    "verify_cleanup",
]

ACTIONS = [
    *SUBGOALS,
    *SHOWCASE_SUBGOALS,
    "inspect_scene",
    "wait",
    "replan",
]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_name: str
    label: str
    instruction: str
    subgoals: tuple[str, ...] = tuple(SUBGOALS)
    horizon_ticks: int = 1000
    requires_disturbance_recovery: bool = True


@dataclass(frozen=True)
class TaskTemplate:
    task_name: str
    label: str
    instruction: str
    subgoals: tuple[str, ...]
    requires_disturbance_recovery: bool = False


TASK_LIBRARY: dict[str, TaskTemplate] = {
    "breakfast_tray": TaskTemplate(
        task_name="breakfast_tray",
        label="Breakfast tray",
        instruction=(
            "Prepare and deliver a breakfast tray. Locate the mug and snack, "
            "clear the workspace, fill the drink, recover from the tray bump, "
            "and deliver without spilling."
        ),
        subgoals=tuple(SUBGOALS),
        requires_disturbance_recovery=True,
    ),
    "spill_recovery": TaskTemplate(
        task_name="spill_recovery",
        label="Spill recovery",
        instruction=(
            "Stabilize a bumped service tray, place an absorbent pad, clear the "
            "workspace, recover the disturbance, and deliver the tray safely."
        ),
        subgoals=(
            "locate_items",
            "stabilize_spill",
            "place_absorbent_pad",
            "clear_workspace",
            "recover_disturbance",
            "deliver_tray",
        ),
        requires_disturbance_recovery=True,
    ),
    "countertop_cleanup": TaskTemplate(
        task_name="countertop_cleanup",
        label="Countertop cleanup",
        instruction=(
            "Complete a countertop cleanup task: scan the workspace, sort "
            "recyclables, place utensils, wipe the countertop, and verify cleanup."
        ),
        subgoals=(
            "scan_countertop",
            "sort_recyclables",
            "place_utensils",
            "wipe_countertop",
            "verify_cleanup",
        ),
        requires_disturbance_recovery=False,
    ),
}


@dataclass(frozen=True)
class SceneConfig:
    mug_position: tuple[float, float, float] = (0.35, -0.2, 0.78)
    snack_position: tuple[float, float, float] = (0.15, 0.3, 0.78)
    tray_position: tuple[float, float, float] = (0.55, 0.0, 0.78)
    disturbance_tick: int = 500
    distractor_count: int = 0
    action_failure_prob: float = 0.0
    disturbance_severity: float = 0.3

    @classmethod
    def from_seed(cls, seed: int) -> "SceneConfig":
        rng = random.Random(seed)
        return cls(
            mug_position=(round(rng.uniform(0.2, 0.45), 3), round(rng.uniform(-0.35, -0.05), 3), 0.78),
            snack_position=(round(rng.uniform(0.05, 0.3), 3), round(rng.uniform(0.15, 0.4), 3), 0.78),
            tray_position=(round(rng.uniform(0.45, 0.65), 3), round(rng.uniform(-0.1, 0.1), 3), 0.78),
            disturbance_tick=rng.choice([350, 500, 650]),
            distractor_count=rng.choice([0, 1, 2, 3]),
            action_failure_prob=rng.choice([0.0, 0.05, 0.1]),
            disturbance_severity=rng.choice([0.2, 0.4, 0.7]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mug_position": self.mug_position,
            "snack_position": self.snack_position,
            "tray_position": self.tray_position,
            "disturbance_tick": self.disturbance_tick,
            "distractor_count": self.distractor_count,
            "action_failure_prob": self.action_failure_prob,
            "disturbance_severity": self.disturbance_severity,
        }


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
    scene: SceneConfig = field(default_factory=SceneConfig)
    task_name: str = "breakfast_tray"
    task: TaskSpec = field(init=False)
    ticks: int = field(init=False, default=0)
    macro_steps: int = field(init=False, default=0)
    progress_index: int = field(init=False, default=0)
    success: bool = field(init=False, default=False)
    done: bool = field(init=False, default=False)
    disturbance_recovered: bool = field(init=False, default=False)
    inspected: bool = field(init=False, default=False)
    replanned: bool = field(init=False, default=False)
    last_failure_reason: str | None = field(init=False, default=None)
    action_history: list[str] = field(init=False, default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        template = TASK_LIBRARY.get(self.task_name)
        if template is None:
            raise ValueError(f"Unknown task_name: {self.task_name!r}")
        self.task = TaskSpec(
            task_id=f"{template.task_name.replace('_', '-')}-{self.seed}",
            task_name=template.task_name,
            label=template.label,
            instruction=template.instruction,
            subgoals=template.subgoals,
            horizon_ticks=self.horizon_ticks,
            requires_disturbance_recovery=template.requires_disturbance_recovery,
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
                str(self.inspected),
                str(self.replanned),
                str(self.scene.as_dict()),
                ",".join(self.action_history[-5:]),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def observe(self) -> dict[str, object]:
        completed = list(self.task.subgoals[: self.progress_index])
        remaining = list(self.task.subgoals[self.progress_index :])
        return {
            "task_id": self.task.task_id,
            "task_name": self.task.task_name,
            "task_label": self.task.label,
            "instruction": self.task.instruction,
            "ticks": self.ticks,
            "macro_steps": self.macro_steps,
            "horizon_ticks": self.horizon_ticks,
            "completed_subgoals": completed,
            "remaining_subgoals": remaining,
            "expected_next": self.expected_action,
            "progress_fraction": self.progress_fraction,
            "disturbance_recovered": self.disturbance_recovered,
            "inspected": self.inspected,
            "replanned": self.replanned,
            "last_failure_reason": self.last_failure_reason,
            "scene": self.scene.as_dict(),
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
        self.last_failure_reason = None

        if action == "inspect_scene":
            self.inspected = True
            progress_delta = 0.0
            reward = -0.01
        elif action == "replan":
            self.replanned = True
            progress_delta = 0.0
            reward = -0.01
        elif action == expected and self._action_can_progress(action):
            self.progress_index += 1
            if action in {"recover_disturbance", "stabilize_spill"}:
                self.disturbance_recovered = True
            progress_delta = self.progress_fraction - before
            reward = 0.25 + progress_delta + (0.1 if self.disturbance_recovered else 0.0)
        else:
            progress_delta = 0.0
            reward = -0.08

        if self.progress_index >= len(self.task.subgoals):
            self.success = (not self.task.requires_disturbance_recovery) or self.disturbance_recovered
            self.done = True
            self.ticks = max(self.ticks, self.horizon_ticks)
            reward += 1.0 if self.success else -0.5
        elif self.macro_steps >= self.max_macro_steps:
            self.done = True
            reward -= 0.25

        return self._transition(action, expected, progress_delta, reward)

    def _action_can_progress(self, action: str) -> bool:
        if action == "locate_items" and self.scene.distractor_count > 0 and not self.inspected:
            self.last_failure_reason = "needs_inspection"
            return False
        if (
            action in {"recover_disturbance", "stabilize_spill"}
            and self.scene.disturbance_severity >= 0.5
            and not self.replanned
        ):
            self.last_failure_reason = "needs_replan"
            return False
        if self.scene.action_failure_prob > 0 and self._rng.random() < self.scene.action_failure_prob:
            self.last_failure_reason = "stochastic_action_failure"
            return False
        return True

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
            if (
                world.expected_action == "locate_items"
                and world.scene.distractor_count > 0
                and not world.inspected
            ):
                return "inspect_scene"
            if (
                world.expected_action in {"recover_disturbance", "stabilize_spill"}
                and world.scene.disturbance_severity >= 0.5
                and not world.replanned
            ):
                return "replan"
            return world.expected_action
        if policy == "random":
            return world._rng.choice(ACTIONS)
        if policy == "fixed_script":
            index = min(world.macro_steps, len(world.task.subgoals) - 1)
            return world.task.subgoals[index]
        if policy == "reactive_script":
            sequence: list[str] = []
            if world.scene.distractor_count > 0:
                sequence.append("inspect_scene")
            for subgoal in world.task.subgoals:
                if subgoal in {"recover_disturbance", "stabilize_spill"} and world.scene.disturbance_severity >= 0.5:
                    sequence.append("replan")
                sequence.append(subgoal)
            index = min(world.macro_steps, len(sequence) - 1)
            return sequence[index]
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
