from __future__ import annotations

from dataclasses import dataclass


ISAAC_ASSET_CANDIDATES: dict[str, tuple[str, ...]] = {
    "mobile_base": (
        "/Isaac/Robots/Clearpath/Jackal/jackal.usd",
        "/Isaac/Robots/Carter/carter_v1.usd",
        "/Isaac/Robots/NVIDIA/Carter/carter.usd",
    ),
    "manipulator": (
        "/Isaac/Robots/Franka/franka_alt_fingers.usd",
        "/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
        "/Isaac/Robots/UR10/ur10.usd",
    ),
    "kitchen_props": (
        "/Isaac/Props/Mugs/mug.usd",
        "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
        "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    ),
}


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float]
    roughness: float = 0.45


@dataclass(frozen=True)
class CameraSpec:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_length: float = 32.0


@dataclass(frozen=True)
class LaneSpec:
    name: str
    label: str
    y_offset: float
    robot_color: str


@dataclass(frozen=True)
class TaskVisualSpec:
    name: str
    label: str
    x_offset: float
    accent_material: str


@dataclass(frozen=True)
class ShowcaseScenePlan:
    title: str
    lanes: tuple[LaneSpec, ...]
    tasks: tuple[TaskVisualSpec, ...]
    materials: tuple[MaterialSpec, ...]
    camera: CameraSpec
    baseline_actions: tuple[str, ...]
    trained_actions: tuple[str, ...]


def make_showcase_scene_plan(
    *,
    baseline_actions: list[str] | tuple[str, ...],
    trained_actions: list[str] | tuple[str, ...],
) -> ShowcaseScenePlan:
    return ShowcaseScenePlan(
        title="RoboCerebra Mobile Service Robot Benchmark",
        lanes=(
            LaneSpec("baseline", "Before training: reactive baseline", -0.9, "alert_orange"),
            LaneSpec("trained", "After training: dense + Gemini vision policy", 0.9, "robot_blue"),
        ),
        tasks=(
            TaskVisualSpec("breakfast_tray", "Breakfast tray", -1.3, "tray_warm"),
            TaskVisualSpec("spill_recovery", "Spill recovery", 0.0, "alert_orange"),
            TaskVisualSpec("countertop_cleanup", "Countertop cleanup", 1.3, "task_green"),
        ),
        materials=(
            MaterialSpec("robot_blue", (0.08, 0.32, 0.85)),
            MaterialSpec("task_green", (0.09, 0.58, 0.27)),
            MaterialSpec("alert_orange", (0.95, 0.42, 0.08)),
            MaterialSpec("tray_warm", (0.93, 0.78, 0.48)),
            MaterialSpec("counter_gray", (0.72, 0.72, 0.68)),
            MaterialSpec("floor_dark", (0.04, 0.05, 0.07)),
            MaterialSpec("object_white", (0.92, 0.92, 0.9)),
        ),
        camera=CameraSpec(position=(2.8, -4.8, 3.2), target=(0.1, 0.0, 0.75)),
        baseline_actions=tuple(baseline_actions),
        trained_actions=tuple(trained_actions),
    )
