from robocerebra_rl.world import BreakfastTrayWorld


def test_expert_policy_completes_a_long_horizon_workflow():
    world = BreakfastTrayWorld(seed=7, horizon_ticks=1000)

    for action in world.expert_actions():
        transition = world.step(action)
        if transition.done:
            break

    assert world.success is True
    assert world.ticks >= 1000
    assert world.progress_fraction == 1.0
    assert world.disturbance_recovered is True


def test_wrong_macro_action_delays_progress_without_ending_episode():
    world = BreakfastTrayWorld(seed=3, horizon_ticks=600)

    transition = world.step("deliver_tray")

    assert transition.done is False
    assert transition.progress_delta == 0.0
    assert transition.reward < 0.0
    assert world.progress_fraction == 0.0
