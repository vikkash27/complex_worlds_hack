from robocerebra_rl.world import ACTIONS, TASK_LIBRARY, BreakfastTrayWorld


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
