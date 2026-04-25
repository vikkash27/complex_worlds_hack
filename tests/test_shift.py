from __future__ import annotations

import pytest

from robocerebra_rl.shift import (
    SHIFT_PROFILES,
    ShiftWorld,
    build_shift_spec,
    expert_shift_actions,
    reactive_shift_actions,
    run_policy_shift,
    shift_spec_from_dict,
)
from robocerebra_rl.world import list_openreward_shift_tasks_for_split


def _run_until_done(world: ShiftWorld, generator):
    for call in generator:
        if world.done:
            break
        method = getattr(world, call["tool"])
        params = call.get("params", {}) or {}
        if isinstance(params, dict):
            method(**params)
        else:
            method(params)


def test_shift_profiles_target_hundreds_to_thousands_of_tool_calls():
    assert SHIFT_PROFILES["train"]["num_jobs"] >= 10
    assert SHIFT_PROFILES["validation"]["num_jobs"] >= 18
    assert SHIFT_PROFILES["test"]["num_jobs"] >= 25


def test_shift_spec_includes_jobs_events_inventory_and_memory():
    spec = build_shift_spec(split="test", seed=4001)

    assert len(spec.jobs) == SHIFT_PROFILES["test"]["num_jobs"]
    assert len(spec.events) == SHIFT_PROFILES["test"]["num_events"]
    assert spec.initial_inventory  # at least one item stocked
    assert spec.initial_memory
    assert spec.max_tool_calls >= 1500


def test_expert_oracle_solves_shift_with_hundreds_of_tool_calls():
    spec = build_shift_spec(split="train", seed=2010)
    world = ShiftWorld(spec=spec)

    _run_until_done(world, expert_shift_actions(world))

    assert world.success is True
    assert world.metrics.tool_calls >= 400
    assert world.metrics.tool_calls <= world.spec.max_tool_calls
    assert world.metrics.events_handled == len(spec.events)
    assert world.metrics.memory_recalls >= 0
    assert len(world.completed_jobs) >= len(spec.jobs)


def test_expert_oracle_on_test_shift_breaks_one_thousand_tool_calls():
    spec = build_shift_spec(split="test", seed=4001)
    world = ShiftWorld(spec=spec)

    _run_until_done(world, expert_shift_actions(world))

    assert world.success is True
    assert world.metrics.tool_calls >= 1000


def test_reactive_baseline_fails_to_handle_events_or_summarize_memory():
    spec = build_shift_spec(split="validation", seed=3001)
    world = ShiftWorld(spec=spec)

    _run_until_done(world, reactive_shift_actions(world))

    # Reactive baseline never acknowledges events or summarizes memory: must fail success.
    assert world.success is False
    assert world.metrics.events_handled == 0
    assert world.summary_completed is False


def test_random_policy_does_not_outperform_expert():
    spec = build_shift_spec(split="train", seed=2003)
    expert_world = ShiftWorld(spec=spec)
    random_world = ShiftWorld(spec=spec)

    expert_metrics = run_policy_shift(expert_world, "expert")
    random_metrics = run_policy_shift(random_world, "random")

    assert expert_metrics["success"] is True
    assert random_metrics["success"] is False
    assert expert_metrics["tool_diversity"] >= random_metrics["tool_diversity"]


def test_openreward_shift_task_list_has_at_least_100_tasks():
    counts = {split: len(list_openreward_shift_tasks_for_split(split)) for split in ("train", "validation", "test")}

    assert counts["train"] >= 70
    assert counts["validation"] >= 16
    assert counts["test"] >= 16
    assert sum(counts.values()) >= 100


def test_shift_spec_round_trips_through_dict():
    spec = build_shift_spec(split="test", seed=4002)
    restored = shift_spec_from_dict(spec.as_dict())

    assert restored.shift_id == spec.shift_id
    assert len(restored.jobs) == len(spec.jobs)
    assert len(restored.events) == len(spec.events)
    assert restored.initial_inventory == spec.initial_inventory


def test_tool_call_budget_fails_shift_when_exceeded():
    spec = build_shift_spec(split="train", seed=2050)
    world = ShiftWorld(spec=spec)

    # Burn the budget on cheap calls.
    for _ in range(spec.max_tool_calls + 5):
        if world.done:
            break
        world.observe()

    assert world.done is True
    assert world.last_failure_reason == "tool_call_budget_exceeded"


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_every_split_has_tractable_expert_runs(split):
    seed = {"train": 2001, "validation": 3001, "test": 4001}[split]
    world = ShiftWorld(spec=build_shift_spec(split=split, seed=seed))

    _run_until_done(world, expert_shift_actions(world))

    assert world.success is True
    assert world.metrics.tool_calls >= 200
