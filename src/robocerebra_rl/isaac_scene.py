from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from robocerebra_rl.isaac_assets import find_unitree_g1_asset, load_unitree_g1_manifest


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
    "humanoid": (
        # Isaac Sim ships G1 under Unitree; see Robot Assets in docs (Create > Isaac > Robots > Unitree > G1).
        "/Isaac/Robots/Unitree/G1/g1.usd",
        "/Isaac/Robots/Unitree/G1/g1_minimal.usd",
        "/Isaac/Robots/Unitree/H1/h1.usd",
        "/Isaac/Robots/Unitree/H1/h1_with_hand.usd",
        "/Isaac/Robots/IsaacSim/Humanoid/humanoid.usd",
        "/Isaac/Robots/IsaacSim/Humanoid/humanoid_instanceable.usd",
    ),
}


def humanoid_asset_candidates() -> tuple[str, ...]:
    """Prefer Unitree G1; allow a local/override USD first when Nucleus paths are locked."""
    base = ISAAC_ASSET_CANDIDATES["humanoid"]
    override = os.environ.get("ROBOCEREBRA_HUMANOID_USD", "").strip()
    if override:
        return (override, *base)
    local = local_unitree_g1_asset()
    if local:
        return (local, *base)
    return base


def local_unitree_g1_asset() -> str | None:
    root = Path(
        os.environ.get(
            "ROBOCEREBRA_UNITREE_ASSET_DIR",
            str(Path(__file__).resolve().parents[2] / "artifacts" / "isaac" / "vendor" / "unitree"),
        )
    )
    manifest_path = root / "unitree_g1_manifest.json"
    try:
        if manifest_path.is_file():
            return str(load_unitree_g1_manifest(manifest_path).asset_path)
        asset = find_unitree_g1_asset(root)
        return str(asset) if asset is not None else None
    except FileNotFoundError:
        return None


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


@dataclass(frozen=True)
class HumanoidShowcaseScenePlan:
    title: str
    humanoid_asset_candidates: tuple[str, ...]
    materials: tuple[MaterialSpec, ...]
    camera: CameraSpec
    baseline_events: tuple[dict[str, object], ...]
    trained_events: tuple[dict[str, object], ...]


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


def make_humanoid_showcase_scene_plan(
    *,
    baseline_events: list[dict[str, object]] | tuple[dict[str, object], ...],
    trained_events: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> HumanoidShowcaseScenePlan:
    return HumanoidShowcaseScenePlan(
        title="RoboCerebra Humanoid OpenReward Showcase",
        humanoid_asset_candidates=humanoid_asset_candidates(),
        materials=(
            MaterialSpec("humanoid_silver", (0.76, 0.80, 0.86), roughness=0.32),
            MaterialSpec("policy_blue", (0.04, 0.22, 0.95), roughness=0.35),
            MaterialSpec("baseline_red", (0.82, 0.12, 0.08), roughness=0.45),
            MaterialSpec("success_green", (0.05, 0.62, 0.26), roughness=0.4),
            MaterialSpec("failure_red", (1.0, 0.05, 0.02), roughness=0.38),
            MaterialSpec("warning_amber", (1.0, 0.64, 0.05), roughness=0.45),
            MaterialSpec("lab_floor", (0.025, 0.030, 0.045), roughness=0.55),
            MaterialSpec("glass_cyan", (0.22, 0.88, 0.95), roughness=0.2),
            MaterialSpec("motion_white", (0.95, 0.98, 1.0), roughness=0.25),
        ),
        camera=CameraSpec(position=(4.6, -6.4, 4.2), target=(0.0, 0.0, 1.25), focal_length=26.0),
        baseline_events=tuple(baseline_events),
        trained_events=tuple(trained_events),
    )
