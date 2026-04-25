from robocerebra_rl.isaac_scene import (
    ISAAC_ASSET_CANDIDATES,
    make_showcase_scene_plan,
)


def test_showcase_scene_plan_has_mobile_robot_assets_and_three_tasks():
    plan = make_showcase_scene_plan(
        baseline_actions=["locate_items", "clear_workspace", "recover_disturbance"],
        trained_actions=["inspect_scene", "locate_items", "clear_workspace", "recover_disturbance"],
    )

    assert plan.title == "RoboCerebra Mobile Service Robot Benchmark"
    assert len(plan.tasks) >= 3
    assert {"Breakfast tray", "Spill recovery", "Countertop cleanup"}.issubset(
        {task.label for task in plan.tasks}
    )
    assert "mobile_base" in ISAAC_ASSET_CANDIDATES
    assert "manipulator" in ISAAC_ASSET_CANDIDATES
    assert plan.lanes[0].label.startswith("Before")
    assert plan.lanes[1].label.startswith("After")


def test_showcase_scene_plan_uses_colorful_materials_and_camera():
    plan = make_showcase_scene_plan(baseline_actions=["wait"], trained_actions=["locate_items"])

    material_names = {material.name for material in plan.materials}
    assert {"robot_blue", "task_green", "alert_orange", "tray_warm"}.issubset(material_names)
    assert plan.camera.position[2] > 2.0
    assert plan.camera.target[0] >= 0.0
