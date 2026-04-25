from robocerebra_rl.isaac_bridge import ISAAC_ACTION_MAP, SceneVariant


def test_isaac_action_map_covers_all_seven_subgoals():
    expected = {
        "locate_items",
        "clear_workspace",
        "pick_mug",
        "fill_drink",
        "place_snack",
        "recover_disturbance",
        "deliver_tray",
    }

    assert expected.issubset(ISAAC_ACTION_MAP)


def test_scene_variant_serializes_modifiable_scene_fields():
    variant = SceneVariant(
        mug_position=(0.35, -0.2, 0.78),
        snack_position=(0.15, 0.3, 0.78),
        disturbance_tick=420,
        distractor_count=3,
    )

    payload = variant.to_task_spec(seed=9)

    assert payload["seed"] == 9
    assert payload["scene"]["disturbance_tick"] == 420
    assert payload["scene"]["distractor_count"] == 3
