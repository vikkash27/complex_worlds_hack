from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.isaac_scene import ISAAC_ASSET_CANDIDATES, ShowcaseScenePlan, make_showcase_scene_plan


def load_actions(path: Path) -> list[str]:
    actions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("tool_name") == "execute_skill" and event.get("action"):
            actions.append(str(event["action"]))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay RoboCerebra traces in Isaac Sim.")
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--trained-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    try:
        from isaacsim import SimulationApp  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.kit import SimulationApp  # type: ignore[import-not-found]

    simulation_app = SimulationApp({"headless": args.headless})

    try:
        from pxr import UsdGeom  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(240)
        stage.SetTimeCodesPerSecond(24)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        baseline = load_actions(args.baseline_trace)
        trained = load_actions(args.trained_trace)
        scene_plan = make_showcase_scene_plan(baseline_actions=baseline, trained_actions=trained)
        materials = _create_materials(stage, scene_plan)
        _add_lighting(stage)
        _add_camera(stage, scene_plan)
        _write_showcase_stage(stage, scene_plan, materials)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        usd_path = args.output_dir / "breakfast_tray_side_by_side.usda"
        stage.GetRootLayer().Export(str(usd_path))
        summary = {
            "baseline_actions": baseline,
            "trained_actions": trained,
            "scene_title": scene_plan.title,
            "tasks": [task.label for task in scene_plan.tasks],
            "asset_candidates": ISAAC_ASSET_CANDIDATES,
            "usd": str(usd_path),
            "note": (
                "Open this USD in Isaac Sim or stream the Brev viewport. "
                "The stage uses colorful proxy geometry with built-in Isaac asset candidates documented here."
            ),
        }
        (args.output_dir / "isaac_replay_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(usd_path)
    finally:
        simulation_app.close()


def _create_materials(stage, scene_plan: ShowcaseScenePlan) -> dict[str, object]:
    from pxr import Sdf, UsdShade  # type: ignore[import-not-found]

    materials: dict[str, object] = {}
    for spec in scene_plan.materials:
        material = UsdShade.Material.Define(stage, f"/World/Materials/{spec.name}")
        shader = UsdShade.Shader.Define(stage, f"/World/Materials/{spec.name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(spec.color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(spec.roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[spec.name] = material
    return materials


def _bind_material(prim, material: object | None) -> None:
    if material is None:
        return
    from pxr import UsdShade  # type: ignore[import-not-found]

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _add_lighting(stage) -> None:
    from pxr import UsdLux  # type: ignore[import-not-found]

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(650.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr(4200.0)
    key.CreateAngleAttr(0.35)


def _add_camera(stage, scene_plan: ShowcaseScenePlan) -> None:
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3d(*scene_plan.camera.position))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(58.0, 0.0, 33.0))
    camera.GetFocalLengthAttr().Set(scene_plan.camera.focal_length)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def _add_cube(
    stage,
    path: str,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: object | None = None,
):
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))
    _bind_material(cube.GetPrim(), material)
    return cube


def _add_asset_reference(
    stage,
    path: str,
    candidates: tuple[str, ...],
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
):
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    prim = stage.DefinePrim(path, "Xform")
    prim.SetCustomDataByKey("fallback", "Colored proxy geometry is shown if referenced Isaac assets do not resolve.")
    for candidate in candidates:
        prim.GetReferences().AddReference(candidate)
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))
    return xformable


def _write_showcase_stage(stage, scene_plan: ShowcaseScenePlan, materials: dict[str, object]) -> None:
    _add_cube(stage, "/World/Floor", (0.0, 0.0, -0.025), (3.2, 2.8, 0.025), materials["floor_dark"])
    for task in scene_plan.tasks:
        _add_task_station(stage, task, materials)
    for lane, actions in zip(scene_plan.lanes, [scene_plan.baseline_actions, scene_plan.trained_actions], strict=True):
        _write_lane(stage, f"/World/{lane.name.title()}", list(actions), lane, materials)


def _add_task_station(stage, task, materials: dict[str, object]) -> None:
    x = task.x_offset
    _add_cube(stage, f"/World/Stations/{task.name}/Counter", (x, 0.0, 0.38), (0.55, 1.18, 0.08), materials["counter_gray"])
    _add_cube(stage, f"/World/Stations/{task.name}/Accent", (x, 0.0, 0.48), (0.5, 1.1, 0.015), materials[task.accent_material])
    _add_cube(
        stage,
        f"/World/Stations/{task.name}/LabelPost",
        (x - 0.48, -1.32, 0.42),
        (0.035, 0.035, 0.38),
        materials[task.accent_material],
    )
    stage.GetPrimAtPath(f"/World/Stations/{task.name}").SetDisplayName(task.label)


def _write_lane(stage, root: str, actions: list[str], lane, materials: dict[str, object]) -> None:
    from pxr import Gf  # type: ignore[import-not-found]

    root_prim = stage.DefinePrim(root, "Xform")
    root_prim.SetDisplayName(lane.label)
    y_offset = lane.y_offset
    _add_cube(stage, f"{root}/LaneRibbon", (0.0, y_offset, 0.02), (3.0, 0.08, 0.02), materials[lane.robot_color])
    base = _add_cube(stage, f"{root}/MobileBase", (-1.3, y_offset, 0.18), (0.22, 0.16, 0.10), materials[lane.robot_color])
    base_asset = _add_asset_reference(
        stage,
        f"{root}/MobileBaseAsset",
        ISAAC_ASSET_CANDIDATES["mobile_base"],
        (-1.3, y_offset, 0.18),
        (0.25, 0.25, 0.25),
    )
    mast = _add_cube(stage, f"{root}/SensorMast", (-1.3, y_offset, 0.48), (0.05, 0.05, 0.25), materials["object_white"])
    arm = _add_cube(stage, f"{root}/ManipulatorArm", (-1.2, y_offset, 0.62), (0.18, 0.035, 0.035), materials[lane.robot_color])
    arm_asset = _add_asset_reference(
        stage,
        f"{root}/ManipulatorAsset",
        ISAAC_ASSET_CANDIDATES["manipulator"],
        (-1.2, y_offset, 0.62),
        (0.12, 0.12, 0.12),
    )
    gripper = _add_cube(stage, f"{root}/Gripper", (-1.05, y_offset, 0.58), (0.055, 0.08, 0.025), materials["object_white"])
    mug = _add_cube(stage, f"{root}/Mug", (-1.35, y_offset - 0.18, 0.58), (0.055, 0.055, 0.09), materials["robot_blue"])
    snack = _add_cube(stage, f"{root}/Snack", (-1.18, y_offset + 0.18, 0.56), (0.09, 0.055, 0.035), materials["tray_warm"])
    tray = _add_cube(stage, f"{root}/Tray", (-1.05, y_offset, 0.53), (0.23, 0.16, 0.025), materials["tray_warm"])
    spill = _add_cube(stage, f"{root}/Spill", (0.02, y_offset + 0.18, 0.51), (0.12, 0.06, 0.008), materials["alert_orange"])
    bin_obj = _add_cube(stage, f"{root}/CleanupBin", (1.3, y_offset - 0.22, 0.55), (0.14, 0.12, 0.12), materials["task_green"])

    x = -1.3
    for index, action in enumerate(actions, start=1):
        frame = index * 24
        if action in {"locate_items", "inspect_scene"}:
            x = -1.15
        elif action in {"clear_workspace", "place_absorbent_pad"}:
            spill.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.24, y_offset + 0.3, 0.51), frame)
            x = -0.25
        elif action in {"pick_mug", "stabilize_spill"}:
            mug.GetOrderedXformOps()[0].Set(Gf.Vec3d(-0.86, y_offset, 0.68), frame)
            x = -0.8
        elif action == "fill_drink":
            x = -0.45
        elif action in {"place_snack", "sort_recyclables", "place_utensils"}:
            snack.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.28, y_offset - 0.12, 0.68), frame)
            x = 1.15
        elif action in {"recover_disturbance", "wipe_countertop"}:
            tray.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.2, y_offset, 0.58), frame)
            x = 0.15
        elif action == "verify_cleanup":
            bin_obj.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.45, y_offset - 0.2, 0.55), frame)
            x = 1.35
        elif action == "deliver_tray":
            tray.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.32, y_offset, 0.58), frame)
            mug.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.32, y_offset - 0.08, 0.68), frame)
            snack.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.32, y_offset + 0.08, 0.63), frame)
            x = 1.3
        base.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 0.18), frame)
        base_asset.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 0.18), frame)
        mast.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 0.48), frame)
        arm.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 0.62), frame)
        arm_asset.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 0.62), frame)
        gripper.GetOrderedXformOps()[0].Set(Gf.Vec3d(x + 0.18, y_offset, 0.58), frame)


if __name__ == "__main__":
    main()
