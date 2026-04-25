from robocerebra_rl.world import (
    ACTIONS,
    TASK_LIBRARY,
    BreakfastTrayWorld,
    iter_policy_actions,
    list_openreward_tasks_for_split,
    openreward_task_dict,
    world_from_task_spec,
)


def test_task_library_exposes_multiple_showcase_tasks():
    assert {"breakfast_tray", "spill_recovery", "countertop_cleanup"}.issubset(TASK_LIBRARY)
    assert "sort_recyclables" in TASK_LIBRARY["countertop_cleanup"].subgoals
    assert "stabilize_spill" in TASK_LIBRARY["spill_recovery"].subgoals
    assert "sort_recyclables" in ACTIONS


def test_non_breakfast_task_can_complete_without_disturbance_recovery_requirement():
    world = BreakfastTrayWorld(seed=4, task_name="countertop_cleanup")

    for action in world.expert_actions():
        transition = world.step(action)
        if transition.done:
            break

    observation = world.observe()
    assert world.success is True
    assert world.progress_fraction == 1.0
    assert observation["task_name"] == "countertop_cleanup"
    assert "countertop cleanup" in observation["instruction"].lower()


def test_openreward_splits_have_expected_task_counts():
    assert len(list_openreward_tasks_for_split("train")) == 19 * 4
    assert len(list_openreward_tasks_for_split("validation")) == 16
    assert len(list_openreward_tasks_for_split("test")) == 16
    total = sum(
        len(list_openreward_tasks_for_split(split)) for split in ("train", "validation", "test")
    )
    assert total > 100


def test_humanoid_openreward_spec_has_enough_macro_budget_for_expert():
    spec = openreward_task_dict("test", 1005, "humanoid_hospitality")
    world = world_from_task_spec(spec)
    assert spec["max_macro_steps"] >= len(TASK_LIBRARY["humanoid_hospitality"].subgoals)
    while not world.done:
        world.step(iter_policy_actions("expert", world))
    assert world.success is True
