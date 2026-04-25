from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.isaac_scene import (
    ISAAC_ASSET_CANDIDATES,
    HumanoidShowcaseScenePlan,
    ShowcaseScenePlan,
    make_humanoid_showcase_scene_plan,
    make_showcase_scene_plan,
)
from robocerebra_rl.humanoid_motion import compile_humanoid_motion, summarize_tool_trace


def load_actions(path: Path) -> list[str]:
    actions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("tool_name") == "execute_skill" and event.get("action"):
            actions.append(str(event["action"]))
    return actions


def load_trace_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def should_use_humanoid_showcase(events: list[dict[str, object]]) -> bool:
    return any(
        (event.get("observation_summary") or {}).get("task_name") == "humanoid_hospitality"
        for event in events
    )


def asset_candidates_custom_data(candidates: tuple[str, ...]) -> str:
    """Serialize asset candidates as a USD-safe scalar custom data value."""
    return json.dumps(list(candidates))


def _isaac_asset_root() -> str | None:
    """Path or omniverse:// URL that prefixes `/Isaac/...` browser paths in running Kit."""
    for mod_name, fn_name in (
        ("isaacsim.core.utils.nucleus", "get_assets_root_path"),
        ("isaacsim.storage.native", "get_assets_root_path"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[fn_name])  # type: ignore[no-untyped-call]
            get_root = getattr(mod, fn_name)
            root = get_root()
            if root:
                return str(root).rstrip("/")
        except Exception:
            pass
    return None


def _pick_usd_reference(
    candidates: tuple[str, ...],
    *,
    label: str,
    prefer_env_override: str | None = "ROBOCEREBRA_HUMANOID_USD",
) -> str:
    """
    Pick a single path for `AddReference`. Prepends Isaac asset root to `/Isaac/...` so
    the robot actually loads in Kit (raw `/Isaac/...` often resolves to nothing in Docker).
    """
    if prefer_env_override:
        ovr = os.environ.get(prefer_env_override, "").strip()
        if ovr:
            p = Path(ovr)
            if p.is_file():
                print(f"[replay] {label} using {prefer_env_override}={p.resolve()}", flush=True)
                return str(p.resolve())
            print(f"[replay] WARNING: {prefer_env_override} is not a file (ignored): {ovr!r}", flush=True)
    base = _isaac_asset_root()
    if base:
        print(f"[replay] Isaac asset root: {base!r}", flush=True)
    else:
        print("[replay] WARNING: get_assets_root_path() unavailable; /Isaac/... references may not load.", flush=True)
    for raw in candidates:
        if not raw:
            continue
        if raw.startswith("/Isaac/") and base:
            resolved = f"{base}{raw}"
        else:
            resolved = raw
        print(f"[replay] {label} AddReference: {raw!r} -> {resolved!r}", flush=True)
        return resolved
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay RoboCerebra traces in Isaac Sim.")
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--trained-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--humanoid-showcase", action="store_true", help="Force the Unitree G1 humanoid long-horizon replay scene.")
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
        _ensure_world_root(stage)
        stage.SetStartTimeCode(0)
        stage.SetTimeCodesPerSecond(24)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        baseline_events = load_trace_events(args.baseline_trace)
        trained_events = load_trace_events(args.trained_trace)
        baseline = [str(event["action"]) for event in baseline_events if event.get("tool_name") == "execute_skill" and event.get("action")]
        trained = [str(event["action"]) for event in trained_events if event.get("tool_name") == "execute_skill" and event.get("action")]
        use_humanoid = args.humanoid_showcase or should_use_humanoid_showcase(baseline_events + trained_events)
        print(
            f"[replay] building {'humanoid' if use_humanoid else 'mobile robot'} scene "
            f"from {len(baseline_events)} baseline events and {len(trained_events)} trained events",
            flush=True,
        )
        scene_plan = (
            make_humanoid_showcase_scene_plan(baseline_events=baseline_events, trained_events=trained_events)
            if use_humanoid
            else make_showcase_scene_plan(baseline_actions=baseline, trained_actions=trained)
        )
        stage.SetEndTimeCode(_end_time_for_scene(scene_plan))
        materials = _create_materials(stage, scene_plan)
        _add_lighting(stage)
        _add_camera(stage, scene_plan)
        if isinstance(scene_plan, HumanoidShowcaseScenePlan):
            _write_humanoid_stage(stage, scene_plan, materials)
        else:
            _write_showcase_stage(stage, scene_plan, materials)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        usd_path = args.output_dir / ("humanoid_openreward_showcase.usda" if use_humanoid else "breakfast_tray_side_by_side.usda")
        export_ok = stage.GetRootLayer().Export(str(usd_path))
        if not export_ok:
            raise RuntimeError(f"Isaac USD export returned false for {usd_path}")
        summary = {
            "baseline_actions": baseline,
            "trained_actions": trained,
            "scene_title": scene_plan.title,
            "tasks": [task.label for task in scene_plan.tasks] if isinstance(scene_plan, ShowcaseScenePlan) else ["Humanoid hospitality lab"],
            "tool_trace_summary": (
                summarize_tool_trace([*scene_plan.baseline_events, *scene_plan.trained_events])
                if isinstance(scene_plan, HumanoidShowcaseScenePlan)
                else {}
            ),
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
        print(f"[replay] wrote {usd_path}", flush=True)
    finally:
        simulation_app.close()


def _end_time_for_scene(scene_plan: ShowcaseScenePlan | HumanoidShowcaseScenePlan) -> int:
    if isinstance(scene_plan, HumanoidShowcaseScenePlan):
        frames = [
            int((event.get("observation_summary") or {}).get("frame_index") or 0)
            for event in [*scene_plan.baseline_events, *scene_plan.trained_events]
        ]
        return max(240, max(frames, default=0) + 24)
    return max(240, max(len(scene_plan.baseline_actions), len(scene_plan.trained_actions)) * 24 + 24)


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


def _ensure_world_root(stage) -> None:
    """Ensure /World exists before we set default prim or define children."""
    from pxr import UsdGeom  # type: ignore[import-not-found]

    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")


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
    *,
    ref_label: str = "asset",
    prefer_env_override: str | None = None,
):
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    prim = stage.DefinePrim(path, "Xform")
    prim.SetCustomDataByKey("fallback", "Colored proxy geometry is shown if referenced Isaac assets do not resolve.")
    prim.SetCustomDataByKey("asset_candidates_json", asset_candidates_custom_data(candidates))
    ref = _pick_usd_reference(candidates, label=ref_label, prefer_env_override=prefer_env_override)
    if ref:
        prim.GetReferences().AddReference(ref)
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


def _write_humanoid_stage(stage, scene_plan: HumanoidShowcaseScenePlan, materials: dict[str, object]) -> None:
    _add_cube(stage, "/World/LabFloor", (0.0, 0.0, -0.025), (4.4, 3.2, 0.025), materials["lab_floor"])
    _add_cube(stage, "/World/MetricWall", (0.0, 1.55, 1.2), (4.1, 0.04, 1.2), materials["glass_cyan"])
    for x, station in [(-1.7, "pantry"), (-0.85, "counter"), (0.0, "sink"), (0.85, "table"), (1.7, "delivery")]:
        stage.DefinePrim(f"/World/Stations/{station}", "Xform").SetDisplayName(station.title())
        _add_cube(stage, f"/World/Stations/{station}/Platform", (x, 0.15, 0.25), (0.36, 0.48, 0.06), materials["humanoid_silver"])
        _add_cube(stage, f"/World/Stations/{station}/Beacon", (x, -0.45, 0.62), (0.045, 0.045, 0.32), materials["warning_amber"])
    _write_humanoid_metrics(stage, scene_plan, materials)
    _write_humanoid_actor(stage, "/World/BaselineHumanoid", scene_plan.baseline_events, -0.78, materials["baseline_red"], scene_plan)
    _write_humanoid_actor(stage, "/World/TrainedHumanoid", scene_plan.trained_events, 0.78, materials["policy_blue"], scene_plan)


def _write_humanoid_metrics(stage, scene_plan: HumanoidShowcaseScenePlan, materials: dict[str, object]) -> None:
    combined = [*scene_plan.baseline_events, *scene_plan.trained_events]
    summary = summarize_tool_trace(combined)
    total = max(1, int(summary["total_tool_calls"]))
    execute = int(summary["execute_skill_calls"])
    score = int(summary["score_progress_calls"])
    _add_cube(stage, "/World/Metrics/ToolCallsTotal", (-1.15, 1.48, 1.75), (min(1.8, total / 160), 0.025, 0.055), materials["success_green"])
    _add_cube(stage, "/World/Metrics/ExecuteSkills", (-1.15, 1.43, 1.58), (min(1.5, execute / 45), 0.025, 0.045), materials["policy_blue"])
    _add_cube(stage, "/World/Metrics/VisionScores", (-1.15, 1.38, 1.43), (min(1.5, score / 45), 0.025, 0.045), materials["warning_amber"])
    tick_count = min(48, total)
    for idx in range(tick_count):
        x = -2.05 + idx * (4.1 / max(1, tick_count - 1))
        material = materials["success_green"] if idx % 4 == 0 else materials["humanoid_silver"]
        _add_cube(stage, f"/World/Metrics/ToolCallRail/Tick_{idx:02d}", (x, -1.42, 0.055), (0.012, 0.035, 0.035), material)
    stations = summary["stations"]
    if isinstance(stations, dict):
        max_station_count = max([int(value) for value in stations.values()], default=1)
        for x, station in [(-1.7, "pantry"), (-0.85, "counter"), (0.0, "sink"), (0.85, "table"), (1.7, "delivery")]:
            height = 0.12 + 0.38 * int(stations.get(station, 0)) / max_station_count
            _add_cube(stage, f"/World/Metrics/StationLoad/{station}", (x, -0.72, height), (0.04, 0.04, height), materials["success_green"])


def _humanoid_root_scale() -> tuple[float, float, float]:
    """Uniform or XYZ scale for the referenced humanoid USD (G1 is authored near real-world size)."""
    raw = os.environ.get("ROBOCEREBRA_HUMANOID_SCALE", "1.0").strip()
    try:
        if "," in raw:
            parts = [float(x) for x in raw.split(",")]
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        v = float(raw)
        return (v, v, v)
    except ValueError:
        return (1.0, 1.0, 1.0)


def _write_humanoid_actor(
    stage,
    root: str,
    events: tuple[dict[str, object], ...],
    y_offset: float,
    material: object,
    scene_plan: HumanoidShowcaseScenePlan,
) -> None:
    from pxr import Gf  # type: ignore[import-not-found]

    # Block proxies were hiding the real referenced robot; G1 is the default asset. Re-enable with ROBOCEREBRA_HUMANOID_PROXY=1.
    use_proxy = os.environ.get("ROBOCEREBRA_HUMANOID_PROXY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    root_prim = stage.DefinePrim(root, "Xform")
    root_prim.SetDisplayName("Baseline humanoid" if y_offset < 0 else "Trained humanoid")
    scale = _humanoid_root_scale()
    humanoid_asset = _add_asset_reference(
        stage,
        f"{root}/HumanoidAsset",
        scene_plan.humanoid_asset_candidates,
        (-1.9, y_offset, 0.85),
        scale,
        ref_label="humanoid (G1)",
        prefer_env_override="ROBOCEREBRA_HUMANOID_USD",
    )
    proxy_prims: list[object] = []
    if use_proxy:
        proxy_prims = [
            _add_cube(stage, f"{root}/ProxyTorso", (-1.9, y_offset, 1.08), (0.12, 0.08, 0.28), material),
            _add_cube(
                stage,
                f"{root}/ProxyHead",
                (-1.9, y_offset, 1.43),
                (0.075, 0.075, 0.075),
                materials_or_default(material),
            ),
            _add_cube(stage, f"{root}/LeftArm", (-1.9, y_offset - 0.12, 1.15), (0.045, 0.035, 0.18), material),
            _add_cube(stage, f"{root}/RightArm", (-1.9, y_offset + 0.12, 1.15), (0.045, 0.035, 0.18), material),
            _add_cube(stage, f"{root}/LeftLeg", (-1.9, y_offset - 0.055, 0.72), (0.045, 0.035, 0.24), material),
            _add_cube(stage, f"{root}/RightLeg", (-1.9, y_offset + 0.055, 0.72), (0.045, 0.035, 0.24), material),
        ]
    left_target = _add_cube(stage, f"{root}/LeftHandTarget", (-1.9, y_offset - 0.16, 1.12), (0.025, 0.025, 0.025), material)
    right_target = _add_cube(stage, f"{root}/RightHandTarget", (-1.9, y_offset + 0.16, 1.12), (0.025, 0.025, 0.025), material)
    tool_token = _add_cube(stage, f"{root}/ToolToken", (-1.9, y_offset, 0.98), (0.06, 0.035, 0.02), material)
    for segment in compile_humanoid_motion(events):
        frame = segment.frame
        x, _, z = segment.root_xyz
        humanoid_asset.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, z), frame)
        left_target.GetOrderedXformOps()[0].Set(
            Gf.Vec3d(segment.left_hand_xyz[0], y_offset - 0.18, segment.left_hand_xyz[2]),
            frame,
        )
        right_target.GetOrderedXformOps()[0].Set(
            Gf.Vec3d(segment.right_hand_xyz[0], y_offset + 0.18, segment.right_hand_xyz[2]),
            frame,
        )
        tool_token.GetOrderedXformOps()[0].Set(
            Gf.Vec3d(segment.right_hand_xyz[0], y_offset, max(0.82, segment.right_hand_xyz[2] - 0.08)),
            frame,
        )
        if use_proxy and proxy_prims:
            proxy_prims[0].GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 1.08), frame)
            proxy_prims[1].GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 1.43), frame)
            proxy_prims[2].GetOrderedXformOps()[0].Set(Gf.Vec3d(x + 0.04, y_offset - 0.15, 1.12), frame)
            proxy_prims[3].GetOrderedXformOps()[0].Set(Gf.Vec3d(x + 0.12, y_offset + 0.15, 1.12), frame)
            proxy_prims[4].GetOrderedXformOps()[0].Set(Gf.Vec3d(x - 0.04, y_offset - 0.055, 0.72), frame)
            proxy_prims[5].GetOrderedXformOps()[0].Set(Gf.Vec3d(x + 0.04, y_offset + 0.055, 0.72), frame)


def materials_or_default(material: object) -> object:
    return material


def _add_task_station(stage, task, materials: dict[str, object]) -> None:
    x = task.x_offset
    stage.DefinePrim(f"/World/Stations/{task.name}", "Xform").SetDisplayName(task.label)
    _add_cube(stage, f"/World/Stations/{task.name}/Counter", (x, 0.0, 0.38), (0.55, 1.18, 0.08), materials["counter_gray"])
    _add_cube(stage, f"/World/Stations/{task.name}/Accent", (x, 0.0, 0.48), (0.5, 1.1, 0.015), materials[task.accent_material])
    _add_cube(
        stage,
        f"/World/Stations/{task.name}/LabelPost",
        (x - 0.48, -1.32, 0.42),
        (0.035, 0.035, 0.38),
        materials[task.accent_material],
    )


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
        ref_label="mobile_base",
    )
    mast = _add_cube(stage, f"{root}/SensorMast", (-1.3, y_offset, 0.48), (0.05, 0.05, 0.25), materials["object_white"])
    arm = _add_cube(stage, f"{root}/ManipulatorArm", (-1.2, y_offset, 0.62), (0.18, 0.035, 0.035), materials[lane.robot_color])
    arm_asset = _add_asset_reference(
        stage,
        f"{root}/ManipulatorAsset",
        ISAAC_ASSET_CANDIDATES["manipulator"],
        (-1.2, y_offset, 0.62),
        (0.12, 0.12, 0.12),
        ref_label="manipulator",
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
