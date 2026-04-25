from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp  # type: ignore

    simulation_app = SimulationApp({"headless": args.headless})

    try:
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        _add_cube(stage, "/World/Table", (0.0, 0.0, 0.35), (1.4, 0.8, 0.08))
        _add_cube(stage, "/World/DeliveryZone", (0.9, 0.0, 0.42), (0.25, 0.5, 0.03))

        baseline = load_actions(args.baseline_trace)
        trained = load_actions(args.trained_trace)
        _write_replay(stage, "/World/Baseline", baseline, y_offset=-0.35)
        _write_replay(stage, "/World/Trained", trained, y_offset=0.35)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        usd_path = args.output_dir / "breakfast_tray_side_by_side.usda"
        stage.GetRootLayer().Export(str(usd_path))
        summary = {
            "baseline_actions": baseline,
            "trained_actions": trained,
            "usd": str(usd_path),
            "note": "Open this USD in Isaac Sim or stream the Brev viewport for the side-by-side 3D demo.",
        }
        (args.output_dir / "isaac_replay_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(usd_path)
    finally:
        simulation_app.close()


def _add_cube(stage, path: str, translate: tuple[float, float, float], scale: tuple[float, float, float]):
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))
    return cube


def _write_replay(stage, root: str, actions: list[str], *, y_offset: float) -> None:
    from pxr import Gf

    arm = _add_cube(stage, f"{root}/RobotArm", (-0.45, y_offset, 0.82), (0.08, 0.08, 0.25))
    mug = _add_cube(stage, f"{root}/Mug", (0.2, y_offset - 0.12, 0.78), (0.06, 0.06, 0.09))
    snack = _add_cube(stage, f"{root}/Snack", (0.1, y_offset + 0.12, 0.78), (0.08, 0.05, 0.03))
    tray = _add_cube(stage, f"{root}/Tray", (0.45, y_offset, 0.74), (0.22, 0.15, 0.025))
    obstacle = _add_cube(stage, f"{root}/Disturbance", (0.35, y_offset + 0.22, 0.78), (0.06, 0.06, 0.06))

    x = -0.45
    for index, action in enumerate(actions, start=1):
        frame = index * 24
        if action == "locate_items":
            x = -0.1
        elif action == "clear_workspace":
            obstacle.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.65, y_offset + 0.25, 0.78), frame)
        elif action == "pick_mug":
            mug.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.34, y_offset, 0.9), frame)
            x = 0.34
        elif action == "fill_drink":
            x = 0.0
        elif action == "place_snack":
            snack.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.45, y_offset + 0.05, 0.82), frame)
            x = 0.45
        elif action == "recover_disturbance":
            tray.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.45, y_offset, 0.78), frame)
            x = 0.5
        elif action == "deliver_tray":
            tray.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.9, y_offset, 0.78), frame)
            mug.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.9, y_offset - 0.05, 0.86), frame)
            snack.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.9, y_offset + 0.05, 0.82), frame)
            x = 0.9
        arm.GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y_offset, 0.9), frame)


if __name__ == "__main__":
    main()
