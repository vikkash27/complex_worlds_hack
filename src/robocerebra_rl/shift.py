"""Long-horizon multi-job shift orchestration for RoboCerebra Reward Lab.

A `ShiftWorld` chains many `BreakfastTrayWorld` jobs into a single episode
(a "service shift") and adds tool-call surface area through tickets,
plans, memory, inventory, a clock, and deterministic non-stationary events.

Design goals:
- Hundreds-to-thousands of tool calls per episode.
- Capabilities that only emerge at long horizons: planning, recall, recovery
  from non-stationarity, resource management.
- Hard but tractable: an expert oracle solves the shift in a bounded number
  of calls; weak baselines stall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Iterable

from robocerebra_rl.world import (
    BreakfastTrayWorld,
    SceneConfig,
    TASK_LIBRARY,
    horizon_ticks_for_seed,
    iter_policy_actions,
    max_macro_steps_for,
    scene_for_split_and_seed,
)


SHIFT_TOOL_NAMES: tuple[str, ...] = (
    # Per-job execution loop (existing tools, now shift-aware).
    "observe",
    "choose_subgoal",
    "execute_skill",
    "score_progress",
    # Shift-only tools.
    "read_ticket",
    "plan_create",
    "plan_revise",
    "memory_write",
    "memory_read",
    "memory_search",
    "memory_summarize",
    "inventory_check",
    "inventory_consume",
    "inventory_restock",
    "clock_get",
    "acknowledge_event",
    "log_job",
    "submit_done",
)


SHIFT_INVENTORY_ITEMS: tuple[str, ...] = (
    "coffee_beans",
    "milk",
    "sugar",
    "mugs",
    "plates",
    "snacks",
    "absorbent_pad",
    "cleaning_spray",
    "towels",
)


SHIFT_GUEST_POOL: tuple[tuple[str, str], ...] = (
    ("guest_amelia", "no_onions"),
    ("guest_brandon", "extra_sugar"),
    ("guest_chen", "lactose_free"),
    ("guest_dora", "gluten_free"),
    ("guest_evan", "decaf_only"),
    ("guest_farah", "no_nuts"),
    ("guest_gus", "vegan"),
    ("guest_henrik", "halal"),
)


SHIFT_PROFILES: dict[str, dict[str, int]] = {
    # Tuned so train shifts are short (fast train), test shifts are long.
    "train": {"num_jobs": 12, "num_events": 3},
    "validation": {"num_jobs": 22, "num_events": 6},
    "test": {"num_jobs": 30, "num_events": 9},
}

SHIFT_TASK_FAMILIES: tuple[str, ...] = (
    "breakfast_tray",
    "spill_recovery",
    "countertop_cleanup",
)


@dataclass(frozen=True)
class ShiftJobSpec:
    job_id: str
    task_name: str
    seed: int
    priority: int
    deadline_tick: int
    required_inventory: tuple[tuple[str, int], ...]
    guest_id: str
    guest_preference: str
    special_instructions: tuple[str, ...]


@dataclass(frozen=True)
class ShiftEventSpec:
    event_id: str
    trigger_after_jobs: int
    kind: str  # 'stockout' | 'vip' | 'preference_recall' | 'spill' | 'time_pressure'
    payload: dict[str, Any]


@dataclass(frozen=True)
class ShiftSpec:
    shift_id: str
    seed: int
    split: str
    jobs: tuple[ShiftJobSpec, ...]
    events: tuple[ShiftEventSpec, ...]
    initial_inventory: tuple[tuple[str, int], ...]
    initial_memory: tuple[tuple[str, str], ...]
    horizon_ticks: int
    max_tool_calls: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "shift_id": self.shift_id,
            "seed": self.seed,
            "split": self.split,
            "horizon_ticks": self.horizon_ticks,
            "max_tool_calls": self.max_tool_calls,
            "num_jobs": len(self.jobs),
            "num_events": len(self.events),
            "initial_inventory": list(self.initial_inventory),
            "initial_memory": list(self.initial_memory),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "task_name": job.task_name,
                    "seed": job.seed,
                    "priority": job.priority,
                    "deadline_tick": job.deadline_tick,
                    "required_inventory": list(job.required_inventory),
                    "guest_id": job.guest_id,
                    "guest_preference": job.guest_preference,
                    "special_instructions": list(job.special_instructions),
                }
                for job in self.jobs
            ],
            "events": [
                {
                    "event_id": event.event_id,
                    "trigger_after_jobs": event.trigger_after_jobs,
                    "kind": event.kind,
                    "payload": dict(event.payload),
                }
                for event in self.events
            ],
        }


def _initial_inventory(rng: random.Random, num_jobs: int) -> tuple[tuple[str, int], ...]:
    """Stock most items but leave at least one item undersupplied to force restock."""
    base_per_job = 2
    inventory: dict[str, int] = {}
    for item in SHIFT_INVENTORY_ITEMS:
        inventory[item] = base_per_job * num_jobs
    short_item = rng.choice(SHIFT_INVENTORY_ITEMS)
    inventory[short_item] = max(1, num_jobs // 4)
    return tuple(sorted(inventory.items()))


def _initial_memory(rng: random.Random, jobs: tuple[ShiftJobSpec, ...]) -> tuple[tuple[str, str], ...]:
    """Pre-populate guest preferences for memory_search drills."""
    memory = {f"guest::{job.guest_id}::preference": job.guest_preference for job in jobs}
    memory["shift::greeting"] = "Welcome to the RoboCerebra hospitality lab"
    memory["shift::policy"] = "always_log_completed_jobs"
    keys = list(memory.keys())
    rng.shuffle(keys)
    return tuple((key, memory[key]) for key in keys)


def _build_jobs(
    *, rng: random.Random, num_jobs: int, base_seed: int, horizon_ticks: int
) -> tuple[ShiftJobSpec, ...]:
    jobs: list[ShiftJobSpec] = []
    horizon_per_job = max(180, horizon_ticks // max(num_jobs, 1))
    for index in range(num_jobs):
        family = SHIFT_TASK_FAMILIES[index % len(SHIFT_TASK_FAMILIES)]
        guest_id, preference = SHIFT_GUEST_POOL[(base_seed + index) % len(SHIFT_GUEST_POOL)]
        priority = 2 if rng.random() < 0.2 else 1
        deadline = horizon_per_job * (index + 1) + rng.choice([60, 90, 120])
        family_template = TASK_LIBRARY[family]
        required = []
        if family == "breakfast_tray":
            required = [("mugs", 1), ("coffee_beans", 1), ("milk", 1), ("snacks", 1)]
        elif family == "spill_recovery":
            required = [("absorbent_pad", 1), ("cleaning_spray", 1), ("towels", 1)]
        elif family == "countertop_cleanup":
            required = [("towels", 1), ("cleaning_spray", 1)]
        instructions: list[str] = []
        if priority == 2:
            instructions.append("VIP_priority")
        if rng.random() < 0.3:
            instructions.append("verify_guest_preference")
        jobs.append(
            ShiftJobSpec(
                job_id=f"job-{index + 1:02d}",
                task_name=family,
                seed=base_seed * 17 + index * 31,
                priority=priority,
                deadline_tick=deadline,
                required_inventory=tuple(required),
                guest_id=guest_id,
                guest_preference=preference,
                special_instructions=tuple(instructions),
            )
        )
        # Quiet the unused-template warning while keeping the dependency wired.
        _ = family_template
    return tuple(jobs)


def _build_events(
    *, rng: random.Random, num_events: int, num_jobs: int, jobs: tuple[ShiftJobSpec, ...]
) -> tuple[ShiftEventSpec, ...]:
    if num_jobs <= 0 or num_events <= 0:
        return ()
    kinds = ("stockout", "vip", "preference_recall", "spill", "time_pressure")
    events: list[ShiftEventSpec] = []
    used_triggers: set[int] = set()
    for index in range(num_events):
        kind = kinds[index % len(kinds)]
        # Spread events across the shift but keep them deterministic.
        trigger = max(1, min(num_jobs - 1, (index + 1) * (num_jobs // (num_events + 1))))
        while trigger in used_triggers and trigger < num_jobs - 1:
            trigger += 1
        used_triggers.add(trigger)
        if kind == "stockout":
            item = SHIFT_INVENTORY_ITEMS[rng.randrange(len(SHIFT_INVENTORY_ITEMS))]
            payload: dict[str, Any] = {"item": item, "shortfall": rng.choice([2, 3, 4])}
        elif kind == "vip":
            target_idx = min(num_jobs - 1, trigger + 1)
            payload = {"job_id": jobs[target_idx].job_id, "raise_priority_to": 2}
        elif kind == "preference_recall":
            target_idx = min(num_jobs - 1, trigger)
            payload = {"guest_id": jobs[target_idx].guest_id, "expected_preference": jobs[target_idx].guest_preference}
        elif kind == "spill":
            payload = {"location": rng.choice(["counter", "tray", "floor"])}
        else:  # time_pressure
            payload = {"deadline_compression_ticks": rng.choice([60, 90, 120])}
        events.append(
            ShiftEventSpec(
                event_id=f"event-{index + 1:02d}-{kind}",
                trigger_after_jobs=trigger,
                kind=kind,
                payload=payload,
            )
        )
    return tuple(events)


def build_shift_spec(*, split: str, seed: int) -> ShiftSpec:
    if split not in SHIFT_PROFILES:
        raise ValueError(f"Unknown split: {split!r}")
    profile = SHIFT_PROFILES[split]
    rng = random.Random(seed * 1000003 + hash(split) % 100003)
    num_jobs = profile["num_jobs"]
    num_events = profile["num_events"]
    horizon_ticks = max(2400, num_jobs * 200)
    jobs = _build_jobs(rng=rng, num_jobs=num_jobs, base_seed=seed, horizon_ticks=horizon_ticks)
    events = _build_events(rng=rng, num_events=num_events, num_jobs=num_jobs, jobs=jobs)
    inventory = _initial_inventory(rng=rng, num_jobs=num_jobs)
    memory = _initial_memory(rng=rng, jobs=jobs)
    # Tool budget: ~50 calls per job + ~25 per event + ~25 setup/close.
    max_tool_calls = num_jobs * 60 + num_events * 30 + 60
    return ShiftSpec(
        shift_id=f"shift-{split}-{seed}",
        seed=seed,
        split=split,
        jobs=jobs,
        events=events,
        initial_inventory=inventory,
        initial_memory=memory,
        horizon_ticks=horizon_ticks,
        max_tool_calls=max_tool_calls,
    )


def shift_spec_from_dict(value: dict[str, Any]) -> ShiftSpec:
    if "shift_id" not in value or "split" not in value or "seed" not in value:
        return build_shift_spec(split=str(value.get("split", "train")), seed=int(value.get("seed", 0)))
    jobs = tuple(
        ShiftJobSpec(
            job_id=str(j["job_id"]),
            task_name=str(j["task_name"]),
            seed=int(j["seed"]),
            priority=int(j["priority"]),
            deadline_tick=int(j["deadline_tick"]),
            required_inventory=tuple((str(item), int(qty)) for item, qty in j.get("required_inventory", [])),
            guest_id=str(j["guest_id"]),
            guest_preference=str(j["guest_preference"]),
            special_instructions=tuple(str(x) for x in j.get("special_instructions", [])),
        )
        for j in value.get("jobs", [])
    )
    events = tuple(
        ShiftEventSpec(
            event_id=str(e["event_id"]),
            trigger_after_jobs=int(e["trigger_after_jobs"]),
            kind=str(e["kind"]),
            payload=dict(e.get("payload", {})),
        )
        for e in value.get("events", [])
    )
    inventory = tuple((str(k), int(v)) for k, v in value.get("initial_inventory", []))
    memory = tuple((str(k), str(v)) for k, v in value.get("initial_memory", []))
    return ShiftSpec(
        shift_id=str(value["shift_id"]),
        seed=int(value["seed"]),
        split=str(value["split"]),
        jobs=jobs,
        events=events,
        initial_inventory=inventory,
        initial_memory=memory,
        horizon_ticks=int(value.get("horizon_ticks", 2400)),
        max_tool_calls=int(value.get("max_tool_calls", 600)),
    )


@dataclass
class ShiftMetrics:
    tool_calls: int = 0
    events_handled: int = 0
    memory_recalls: int = 0
    memory_writes: int = 0
    plan_revisions: int = 0
    inventory_restocks: int = 0
    score_progress_calls: int = 0
    tool_call_log: list[str] = field(default_factory=list)
    completed_jobs: list[str] = field(default_factory=list)

    def record(self, name: str) -> None:
        self.tool_calls += 1
        self.tool_call_log.append(name)


@dataclass
class ShiftWorld:
    spec: ShiftSpec

    inventory: dict[str, int] = field(init=False)
    memory: dict[str, str] = field(init=False)
    completed_jobs: list[str] = field(init=False, default_factory=list)
    failed_jobs: list[str] = field(init=False, default_factory=list)
    current_job_idx: int = field(init=False, default=0)
    current_world: BreakfastTrayWorld | None = field(init=False, default=None)
    current_ticket_read: bool = field(init=False, default=False)
    current_plan: tuple[str, ...] | None = field(init=False, default=None)
    current_plan_satisfies_inventory: bool = field(init=False, default=False)
    pending_events: list[ShiftEventSpec] = field(init=False)
    active_events: list[str] = field(init=False, default_factory=list)
    acknowledged_events: list[str] = field(init=False, default_factory=list)
    pending_event_resolution: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    deadline_compression: int = field(init=False, default=0)
    summary_completed: bool = field(init=False, default=False)
    submitted: bool = field(init=False, default=False)
    success: bool = field(init=False, default=False)
    done: bool = field(init=False, default=False)
    last_failure_reason: str | None = field(init=False, default=None)
    metrics: ShiftMetrics = field(init=False, default_factory=ShiftMetrics)
    ticks: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.inventory = dict(self.spec.initial_inventory)
        self.memory = dict(self.spec.initial_memory)
        self.pending_events = list(self.spec.events)
        self._materialize_current_job()

    # ----- helpers --------------------------------------------------------

    @property
    def current_job(self) -> ShiftJobSpec | None:
        if self.current_job_idx >= len(self.spec.jobs):
            return None
        return self.spec.jobs[self.current_job_idx]

    @property
    def remaining_jobs(self) -> int:
        return max(0, len(self.spec.jobs) - self.current_job_idx)

    def _materialize_current_job(self) -> None:
        job = self.current_job
        if job is None:
            self.current_world = None
            return
        scene = scene_for_split_and_seed(job.seed)
        # Tighten inventory/disturbance demands so per-job worlds match family expectations.
        scene = self._customize_scene(job, scene)
        self.current_world = BreakfastTrayWorld(
            seed=job.seed,
            horizon_ticks=horizon_ticks_for_seed(job.seed),
            max_macro_steps=max_macro_steps_for(job.task_name, job.seed),
            scene=scene,
            task_name=job.task_name,
        )
        self.current_ticket_read = False
        self.current_plan = None
        self.current_plan_satisfies_inventory = False

    @staticmethod
    def _customize_scene(job: ShiftJobSpec, scene: SceneConfig) -> SceneConfig:
        if job.priority == 2:
            return SceneConfig(
                mug_position=scene.mug_position,
                snack_position=scene.snack_position,
                tray_position=scene.tray_position,
                disturbance_tick=scene.disturbance_tick,
                distractor_count=max(2, scene.distractor_count),
                action_failure_prob=max(0.1, scene.action_failure_prob),
                disturbance_severity=max(0.5, scene.disturbance_severity),
            )
        return scene

    def _trigger_due_events(self) -> None:
        # Any event whose trigger threshold is <= number of completed jobs becomes active.
        ready: list[ShiftEventSpec] = []
        remaining: list[ShiftEventSpec] = []
        for event in self.pending_events:
            if event.trigger_after_jobs <= len(self.completed_jobs) and event.event_id not in self.acknowledged_events:
                ready.append(event)
            else:
                remaining.append(event)
        for event in ready:
            if event.event_id not in self.active_events:
                self.active_events.append(event.event_id)
            self.pending_event_resolution.setdefault(event.event_id, {"kind": event.kind, "payload": dict(event.payload)})
        self.pending_events = remaining

    def _event_by_id(self, event_id: str) -> ShiftEventSpec | None:
        for event in self.spec.events:
            if event.event_id == event_id:
                return event
        return None

    def state_hash(self) -> str:
        raw = "|".join(
            [
                self.spec.shift_id,
                str(self.current_job_idx),
                ",".join(self.completed_jobs[-3:]),
                ",".join(self.active_events[-3:]),
                ",".join(self.acknowledged_events[-3:]),
                str(self.metrics.tool_calls),
                str(sorted(self.inventory.items())[:3]),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def overview(self) -> dict[str, Any]:
        self._trigger_due_events()
        job = self.current_job
        return {
            "shift_id": self.spec.shift_id,
            "split": self.spec.split,
            "tool_calls": self.metrics.tool_calls,
            "max_tool_calls": self.spec.max_tool_calls,
            "completed_jobs": list(self.completed_jobs),
            "remaining_jobs": self.remaining_jobs,
            "current_job": job.job_id if job else None,
            "current_task_name": job.task_name if job else None,
            "current_ticket_read": self.current_ticket_read,
            "current_plan_registered": self.current_plan is not None,
            "current_plan_satisfies_inventory": self.current_plan_satisfies_inventory,
            "active_events": list(self.active_events),
            "acknowledged_events": list(self.acknowledged_events),
            "events_handled": self.metrics.events_handled,
            "memory_keys": sorted(self.memory.keys()),
            "memory_writes": self.metrics.memory_writes,
            "memory_recalls": self.metrics.memory_recalls,
            "plan_revisions": self.metrics.plan_revisions,
            "inventory_restocks": self.metrics.inventory_restocks,
            "score_progress_calls": self.metrics.score_progress_calls,
            "deadline_compression": self.deadline_compression,
            "summary_completed": self.summary_completed,
            "submitted": self.submitted,
            "success": self.success,
            "done": self.done,
            "last_failure_reason": self.last_failure_reason,
            "ticks": self.ticks,
        }

    # ----- tool implementations -----------------------------------------

    def _record(self, tool: str) -> None:
        self.metrics.record(tool)
        if self.metrics.tool_calls > self.spec.max_tool_calls:
            self.done = True
            self.last_failure_reason = "tool_call_budget_exceeded"

    def observe(self) -> dict[str, Any]:
        self._record("observe")
        overview = self.overview()
        if self.current_world is not None:
            overview["current_subgoals_remaining"] = list(self.current_world.task.subgoals[self.current_world.progress_index:])
            overview["current_subgoals_completed"] = list(self.current_world.task.subgoals[: self.current_world.progress_index])
            overview["current_expected_next"] = self.current_world.expected_action
        return overview

    def read_ticket(self) -> dict[str, Any]:
        self._record("read_ticket")
        job = self.current_job
        if job is None:
            self.last_failure_reason = "no_more_tickets"
            return {"ticket": None, "remaining_jobs": 0}
        self.current_ticket_read = True
        guest_pref = self.memory.get(f"guest::{job.guest_id}::preference", job.guest_preference)
        return {
            "ticket": {
                "job_id": job.job_id,
                "task_name": job.task_name,
                "label": TASK_LIBRARY[job.task_name].label,
                "instruction": TASK_LIBRARY[job.task_name].instruction,
                "priority": job.priority,
                "deadline_tick": max(60, job.deadline_tick - self.deadline_compression),
                "required_inventory": list(job.required_inventory),
                "guest_id": job.guest_id,
                "guest_preference": guest_pref,
                "special_instructions": list(job.special_instructions),
                "subgoals": list(TASK_LIBRARY[job.task_name].subgoals),
            },
        }

    def plan_create(self, steps: Iterable[str]) -> dict[str, Any]:
        self._record("plan_create")
        steps_t = tuple(str(s) for s in steps)
        if not self.current_ticket_read:
            self.last_failure_reason = "plan_before_ticket"
            return {"accepted": False, "reason": "plan_before_ticket"}
        if not steps_t:
            self.last_failure_reason = "empty_plan"
            return {"accepted": False, "reason": "empty_plan"}
        self.current_plan = steps_t
        self.current_plan_satisfies_inventory = self._plan_covers_inventory()
        return {"accepted": True, "plan": list(steps_t), "covers_inventory": self.current_plan_satisfies_inventory}

    def plan_revise(self, steps: Iterable[str]) -> dict[str, Any]:
        self._record("plan_revise")
        steps_t = tuple(str(s) for s in steps)
        if self.current_plan is None:
            self.last_failure_reason = "revise_before_plan"
            return {"accepted": False, "reason": "revise_before_plan"}
        self.current_plan = steps_t
        self.metrics.plan_revisions += 1
        self.current_plan_satisfies_inventory = self._plan_covers_inventory()
        return {"accepted": True, "plan": list(steps_t), "plan_revisions": self.metrics.plan_revisions}

    def _plan_covers_inventory(self) -> bool:
        job = self.current_job
        if job is None or self.current_plan is None:
            return False
        plan_str = " ".join(self.current_plan).lower()
        return all(item in plan_str for item, qty in job.required_inventory if qty > 0)

    def memory_write(self, key: str, value: str) -> dict[str, Any]:
        self._record("memory_write")
        self.memory[str(key)] = str(value)
        self.metrics.memory_writes += 1
        return {"accepted": True, "key": key, "size": len(self.memory)}

    def memory_read(self, key: str) -> dict[str, Any]:
        self._record("memory_read")
        value = self.memory.get(str(key))
        if value is not None:
            self.metrics.memory_recalls += 1
        return {"key": key, "value": value, "found": value is not None}

    def memory_search(self, query: str) -> dict[str, Any]:
        self._record("memory_search")
        q = str(query).lower()
        hits: list[dict[str, str]] = []
        for key, value in self.memory.items():
            if q and (q in key.lower() or q in value.lower()):
                hits.append({"key": key, "value": value})
        if hits:
            self.metrics.memory_recalls += 1
        return {"query": query, "hits": hits[:10]}

    def memory_summarize(self) -> dict[str, Any]:
        self._record("memory_summarize")
        guest_count = sum(1 for k in self.memory if k.startswith("guest::"))
        log_count = sum(1 for k in self.memory if k.startswith("job::"))
        unique_guests = len({j.guest_id for j in self.spec.jobs})
        summary = {
            "shift_id": self.spec.shift_id,
            "completed_jobs": list(self.completed_jobs),
            "active_events": list(self.active_events),
            "guests_known": guest_count,
            "jobs_logged": log_count,
            "unique_guests_required": unique_guests,
            "summary_size": len(self.memory),
        }
        adequate = (
            log_count >= len(self.completed_jobs)
            and guest_count >= unique_guests
            and len(self.completed_jobs) >= 1
        )
        if adequate:
            self.summary_completed = True
        return {"summary": summary, "adequate": adequate}

    def inventory_check(self, item: str | None = None) -> dict[str, Any]:
        self._record("inventory_check")
        if item is None:
            return {"inventory": dict(sorted(self.inventory.items()))}
        return {"item": item, "qty": int(self.inventory.get(item, 0))}

    def inventory_consume(self, item: str, qty: int = 1) -> dict[str, Any]:
        self._record("inventory_consume")
        item_s = str(item)
        qty_i = int(qty)
        have = int(self.inventory.get(item_s, 0))
        if have < qty_i:
            self.last_failure_reason = "inventory_short"
            return {"consumed": False, "item": item_s, "have": have, "needed": qty_i}
        self.inventory[item_s] = have - qty_i
        return {"consumed": True, "item": item_s, "remaining": self.inventory[item_s]}

    def inventory_restock(self, item: str, qty: int = 2) -> dict[str, Any]:
        self._record("inventory_restock")
        item_s = str(item)
        qty_i = max(1, int(qty))
        self.inventory[item_s] = int(self.inventory.get(item_s, 0)) + qty_i
        self.metrics.inventory_restocks += 1
        return {"item": item_s, "qty_added": qty_i, "total": self.inventory[item_s]}

    def clock_get(self) -> dict[str, Any]:
        self._record("clock_get")
        ticks_used = self.ticks + (self.current_world.ticks if self.current_world else 0)
        return {
            "ticks": ticks_used,
            "horizon_ticks": self.spec.horizon_ticks,
            "deadline_compression": self.deadline_compression,
            "fraction_used": round(ticks_used / max(1, self.spec.horizon_ticks), 6),
        }

    def acknowledge_event(self, event_id: str) -> dict[str, Any]:
        self._record("acknowledge_event")
        if event_id not in self.active_events:
            self.last_failure_reason = "unknown_event"
            return {"accepted": False, "reason": "unknown_event"}
        self.active_events.remove(event_id)
        self.acknowledged_events.append(event_id)
        self.metrics.events_handled += 1
        spec = self._event_by_id(event_id)
        if spec is None:
            return {"accepted": True, "kind": "unknown"}
        if spec.kind == "stockout":
            item = str(spec.payload.get("item"))
            shortfall = int(spec.payload.get("shortfall", 0))
            self.inventory[item] = max(0, int(self.inventory.get(item, 0)) - shortfall)
        elif spec.kind == "vip":
            target_id = str(spec.payload.get("job_id"))
            for idx, job in enumerate(self.spec.jobs):
                if job.job_id == target_id and idx >= self.current_job_idx:
                    # Promote priority by replacing the spec with a higher-priority copy.
                    promoted = ShiftJobSpec(
                        job_id=job.job_id,
                        task_name=job.task_name,
                        seed=job.seed,
                        priority=2,
                        deadline_tick=job.deadline_tick,
                        required_inventory=job.required_inventory,
                        guest_id=job.guest_id,
                        guest_preference=job.guest_preference,
                        special_instructions=job.special_instructions + ("VIP_priority",),
                    )
                    new_jobs = list(self.spec.jobs)
                    new_jobs[idx] = promoted
                    object.__setattr__(self.spec, "jobs", tuple(new_jobs))
                    if idx == self.current_job_idx:
                        self._materialize_current_job()
                    break
        elif spec.kind == "preference_recall":
            # Caller still has to look up via memory_search; acknowledging only opens it.
            pass
        elif spec.kind == "spill":
            # Insert an extra spill_recovery job right after the current one.
            location = str(spec.payload.get("location"))
            extra_seed = self.spec.seed * 7919 + len(self.acknowledged_events)
            extra = ShiftJobSpec(
                job_id=f"spill-{event_id}",
                task_name="spill_recovery",
                seed=extra_seed,
                priority=2,
                deadline_tick=self.spec.horizon_ticks,
                required_inventory=(("absorbent_pad", 1), ("cleaning_spray", 1), ("towels", 1)),
                guest_id="guest_amelia",
                guest_preference=f"clean_{location}",
                special_instructions=("incident_response", f"location_{location}"),
            )
            new_jobs = list(self.spec.jobs)
            insert_at = min(self.current_job_idx + 1, len(new_jobs))
            new_jobs.insert(insert_at, extra)
            object.__setattr__(self.spec, "jobs", tuple(new_jobs))
        elif spec.kind == "time_pressure":
            self.deadline_compression += int(spec.payload.get("deadline_compression_ticks", 0))
        return {"accepted": True, "kind": spec.kind, "events_handled": self.metrics.events_handled}

    def choose_subgoal(self, subgoal: str) -> dict[str, Any]:
        self._record("choose_subgoal")
        if self.current_world is None:
            self.last_failure_reason = "no_active_job"
            return {"accepted": False, "reason": "no_active_job"}
        self.current_world.action_history.append(f"choose:{subgoal}")
        return {"accepted": True, "subgoal": subgoal, "expected_next": self.current_world.expected_action}

    def execute_skill(self, action: str) -> dict[str, Any]:
        self._record("execute_skill")
        if self.current_world is None:
            self.last_failure_reason = "no_active_job"
            return {"accepted": False, "reason": "no_active_job", "reward": 0.0, "progress_delta": 0.0}
        if not self.current_ticket_read:
            self.last_failure_reason = "execute_before_ticket"
            return {"accepted": False, "reason": "execute_before_ticket", "reward": -0.05, "progress_delta": 0.0}
        if self.current_plan is None:
            self.last_failure_reason = "execute_before_plan"
            return {"accepted": False, "reason": "execute_before_plan", "reward": -0.05, "progress_delta": 0.0}
        transition = self.current_world.step(action)
        return {
            "accepted": True,
            "action": action,
            "reward": transition.reward,
            "progress_delta": transition.progress_delta,
            "progress_after": transition.progress_after,
            "done": transition.done,
            "success": transition.success,
            "expected_action": transition.expected_action,
        }

    def score_progress(self, subgoal: str) -> dict[str, Any]:
        self._record("score_progress")
        self.metrics.score_progress_calls += 1
        if self.current_world is None:
            return {"progress_delta": 0.0, "confidence": 0.0, "subgoal": subgoal, "rationale": "no_active_job"}
        # Use the simulator's symbolic progress as the "VLM proxy".
        completed = self.current_world.progress_index
        total = len(self.current_world.task.subgoals)
        progress_after = round(completed / max(1, total), 6)
        return {
            "progress_delta": progress_after,
            "confidence": 0.65,
            "subgoal": subgoal,
            "rationale": "Symbolic shift-VLM proxy (set ROBOCEREBRA_USE_GEMINI_VISION=1 for live).",
        }

    def log_job(self, summary: str) -> dict[str, Any]:
        self._record("log_job")
        job = self.current_job
        if job is None or self.current_world is None:
            self.last_failure_reason = "log_without_job"
            return {"accepted": False, "reason": "log_without_job"}
        if not self.current_world.done:
            # Allow logging even if execution stalled; mark the job as failed.
            self.failed_jobs.append(job.job_id)
            outcome = "failed"
        else:
            self.completed_jobs.append(job.job_id)
            outcome = "completed" if self.current_world.success else "completed_with_issues"
        self.memory[f"job::{job.job_id}::summary"] = str(summary)
        self.ticks += self.current_world.ticks if self.current_world else 0
        self.current_job_idx += 1
        self._trigger_due_events()
        self._materialize_current_job()
        return {
            "accepted": True,
            "outcome": outcome,
            "completed_jobs": len(self.completed_jobs),
            "remaining_jobs": self.remaining_jobs,
        }

    def submit_done(self) -> dict[str, Any]:
        self._record("submit_done")
        self.submitted = True
        self.done = True
        all_jobs_advanced = self.current_job_idx >= len(self.spec.jobs)
        all_events_acked = not self.active_events
        good_summary = self.summary_completed
        # Success requires ALL three: completing every job, handling every event, and summarizing memory.
        self.success = all_jobs_advanced and all_events_acked and good_summary
        return {
            "submitted": True,
            "success": self.success,
            "all_jobs_advanced": all_jobs_advanced,
            "all_events_acked": all_events_acked,
            "summary_completed": good_summary,
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "tool_calls": self.metrics.tool_calls,
        }


# ---------------------------------------------------------------------------
# Expert oracle for ShiftWorld -- proves tractability and produces dense traces.
# ---------------------------------------------------------------------------


def expert_shift_actions(world: ShiftWorld) -> Iterable[dict[str, Any]]:
    """Yield (tool, params) dicts that solve the shift.

    Designed so a hosted client can replay it call-by-call. Each item is a
    dict with `{"tool": <name>, "params": <dict>}`.
    """

    yield {"tool": "observe", "params": {}}
    yield {"tool": "memory_summarize", "params": {}}

    while True:
        if world.done:
            break
        if world.current_job is None:
            break
        job = world.current_job

        # Always read ticket before plan/execute.
        yield {"tool": "read_ticket", "params": {}}
        # If a preference_recall event is active for this guest, recall it via memory_search.
        for event_id in list(world.active_events):
            spec = world._event_by_id(event_id)
            if spec is None:
                continue
            if spec.kind == "preference_recall":
                yield {"tool": "memory_search", "params": {"query": spec.payload.get("guest_id", job.guest_id)}}
                yield {"tool": "acknowledge_event", "params": {"event_id": event_id}}
            elif spec.kind == "stockout":
                item = str(spec.payload.get("item"))
                yield {"tool": "inventory_restock", "params": {"item": item, "qty": int(spec.payload.get("shortfall", 2)) + 1}}
                yield {"tool": "acknowledge_event", "params": {"event_id": event_id}}
            elif spec.kind == "vip":
                yield {"tool": "memory_write", "params": {"key": f"event::{event_id}::handled", "value": "vip_priority_set"}}
                yield {"tool": "acknowledge_event", "params": {"event_id": event_id}}
                yield {"tool": "plan_revise", "params": {"steps": [f"prioritize::{spec.payload.get('job_id')}"]}}
            elif spec.kind == "spill":
                yield {"tool": "memory_write", "params": {"key": f"event::{event_id}::handled", "value": "spill_logged"}}
                yield {"tool": "acknowledge_event", "params": {"event_id": event_id}}
            elif spec.kind == "time_pressure":
                yield {"tool": "clock_get", "params": {}}
                yield {"tool": "plan_revise", "params": {"steps": ["accelerate"]}}
                yield {"tool": "acknowledge_event", "params": {"event_id": event_id}}

        # Build a plan that names the inventory items so plan_create accepts it.
        plan_steps = [f"acquire::{item}" for item, qty in job.required_inventory if qty > 0]
        plan_steps += [f"execute::{sub}" for sub in TASK_LIBRARY[job.task_name].subgoals]
        plan_steps.append(f"log::{job.job_id}")
        yield {"tool": "plan_create", "params": {"steps": plan_steps}}
        yield {"tool": "inventory_check", "params": {}}
        for item, qty in job.required_inventory:
            if qty <= 0:
                continue
            available = int(world.inventory.get(item, 0))
            if available < qty:
                yield {"tool": "inventory_restock", "params": {"item": item, "qty": qty - available + 2}}
            yield {"tool": "inventory_consume", "params": {"item": item, "qty": qty}}

        # Per-subgoal loop: observe -> choose -> execute -> score -> memory_write.
        # Use the per-job expert from world.iter_policy_actions for action selection.
        sub_world = world.current_world
        if sub_world is None:
            break
        max_iters = max(20, sub_world.max_macro_steps + 5)
        iters = 0
        while not sub_world.done and iters < max_iters:
            iters += 1
            yield {"tool": "observe", "params": {}}
            expected = sub_world.expected_action
            yield {"tool": "choose_subgoal", "params": {"subgoal": expected}}
            action = iter_policy_actions("expert", sub_world)
            yield {"tool": "execute_skill", "params": {"action": action}}
            yield {"tool": "score_progress", "params": {"subgoal": expected}}
            yield {
                "tool": "memory_write",
                "params": {
                    "key": f"job::{job.job_id}::step::{sub_world.macro_steps:02d}",
                    "value": f"{action}->{sub_world.progress_fraction:.3f}",
                },
            }
        yield {"tool": "memory_write", "params": {"key": f"job::{job.job_id}::completed", "value": "true"}}
        yield {"tool": "log_job", "params": {"summary": f"{job.task_name} for {job.guest_id} done"}}

    # Drain remaining events that fired after the last log_job (deterministic edge case).
    for event_id in list(world.active_events):
        spec = world._event_by_id(event_id)
        if spec is None:
            continue
        if spec.kind == "stockout":
            yield {"tool": "inventory_restock", "params": {"item": spec.payload.get("item"), "qty": 2}}
        elif spec.kind == "preference_recall":
            yield {"tool": "memory_search", "params": {"query": spec.payload.get("guest_id", "guest")}}
        elif spec.kind == "time_pressure":
            yield {"tool": "clock_get", "params": {}}
        yield {"tool": "acknowledge_event", "params": {"event_id": event_id}}
    yield {"tool": "memory_summarize", "params": {}}
    yield {"tool": "submit_done", "params": {}}


def run_expert_shift(world: ShiftWorld) -> dict[str, Any]:
    """Run the expert oracle to completion, returning shift metrics."""
    for call in expert_shift_actions(world):
        if world.done:
            break
        tool = call["tool"]
        params = call.get("params", {})
        method = getattr(world, tool)
        if isinstance(params, dict):
            method(**params)
        else:
            method(params)
        if world.done:
            break
    return {
        "tool_calls": world.metrics.tool_calls,
        "events_handled": world.metrics.events_handled,
        "memory_recalls": world.metrics.memory_recalls,
        "plan_revisions": world.metrics.plan_revisions,
        "inventory_restocks": world.metrics.inventory_restocks,
        "score_progress_calls": world.metrics.score_progress_calls,
        "completed_jobs": len(world.completed_jobs),
        "failed_jobs": len(world.failed_jobs),
        "success": world.success,
        "tool_diversity": _tool_diversity(world.metrics.tool_call_log),
    }


def _tool_diversity(tool_log: list[str]) -> int:
    return len(set(tool_log))


# ---------------------------------------------------------------------------
# Lightweight per-call policy adapters used by reactive baselines.
# ---------------------------------------------------------------------------


def reactive_shift_actions(world: ShiftWorld) -> Iterable[dict[str, Any]]:
    """A weak baseline that does NOT use memory or proper plan revision.

    It tries to push through jobs directly with the existing reactive_script
    per-job policy and ignores events. Designed to fail to satisfy the
    `submit_done` success criteria (events / summary).
    """
    yield {"tool": "observe", "params": {}}
    while not world.done and world.current_job is not None:
        job = world.current_job
        yield {"tool": "read_ticket", "params": {}}
        plan_steps = [f"execute::{sub}" for sub in TASK_LIBRARY[job.task_name].subgoals]
        yield {"tool": "plan_create", "params": {"steps": plan_steps}}
        sub_world = world.current_world
        if sub_world is None:
            break
        iters = 0
        while not sub_world.done and iters < sub_world.max_macro_steps + 2:
            iters += 1
            action = iter_policy_actions("reactive_script", sub_world)
            yield {"tool": "choose_subgoal", "params": {"subgoal": sub_world.expected_action}}
            yield {"tool": "execute_skill", "params": {"action": action}}
        yield {"tool": "log_job", "params": {"summary": "reactive"}}
    yield {"tool": "submit_done", "params": {}}


def random_shift_actions(world: ShiftWorld, *, seed: int = 0) -> Iterable[dict[str, Any]]:
    rng = random.Random(seed)
    pool = ("observe", "read_ticket", "plan_create", "execute_skill", "log_job", "submit_done")
    while not world.done and world.metrics.tool_calls < world.spec.max_tool_calls:
        choice = rng.choice(pool)
        if choice == "observe":
            yield {"tool": "observe", "params": {}}
        elif choice == "read_ticket":
            yield {"tool": "read_ticket", "params": {}}
        elif choice == "plan_create":
            yield {"tool": "plan_create", "params": {"steps": ["random::step"]}}
        elif choice == "execute_skill":
            yield {"tool": "execute_skill", "params": {"action": "wait"}}
        elif choice == "log_job":
            yield {"tool": "log_job", "params": {"summary": "random"}}
        elif choice == "submit_done":
            yield {"tool": "submit_done", "params": {}}
            return


def run_policy_shift(world: ShiftWorld, policy: str) -> dict[str, Any]:
    if policy == "expert":
        gen = expert_shift_actions(world)
    elif policy == "reactive_script":
        gen = reactive_shift_actions(world)
    elif policy == "random":
        gen = random_shift_actions(world, seed=world.spec.seed)
    else:
        raise ValueError(f"Unknown shift policy: {policy!r}")
    for call in gen:
        if world.done:
            break
        tool = call["tool"]
        params = call.get("params", {})
        method = getattr(world, tool)
        if isinstance(params, dict):
            method(**params)
        else:
            method(params)
    return {
        "tool_calls": world.metrics.tool_calls,
        "events_handled": world.metrics.events_handled,
        "memory_recalls": world.metrics.memory_recalls,
        "plan_revisions": world.metrics.plan_revisions,
        "inventory_restocks": world.metrics.inventory_restocks,
        "score_progress_calls": world.metrics.score_progress_calls,
        "completed_jobs": len(world.completed_jobs),
        "failed_jobs": len(world.failed_jobs),
        "success": world.success,
        "tool_diversity": _tool_diversity(world.metrics.tool_call_log),
    }
