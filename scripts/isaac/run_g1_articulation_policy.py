from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.g1_articulation_policy import build_g1_policy_frame, select_supported_g1_joints  # noqa: E402
from robocerebra_rl.humanoid_motion import compile_humanoid_motion  # noqa: E402
from robocerebra_rl.isaac_assets import load_unitree_g1_manifest  # noqa: E402


def load_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test articulated Unitree G1 policy playback in Isaac Sim.")
    parser.add_argument("--trace", type=Path, default=ROOT / "artifacts" / "traces" / "humanoid_trained_long_horizon.jsonl")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "isaac" / "vendor" / "unitree" / "unitree_g1_manifest.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "isaac" / "g1_articulation_policy.json")
    args = parser.parse_args()

    manifest = load_unitree_g1_manifest(args.manifest)
    if manifest.asset_kind != "usd":
        raise RuntimeError(
            f"Articulation playback needs a USD asset, got {manifest.asset_kind!r} at {manifest.asset_path}. "
            "Fetch Unitree USD assets via scripts/isaac/fetch_unitree_g1_assets.py."
        )

    try:
        from isaacsim import SimulationApp  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.kit import SimulationApp  # type: ignore[import-not-found]

    simulation_app = SimulationApp({"headless": True})
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        robot = stage.DefinePrim("/World/UnitreeG1", "Xform")
        robot.GetReferences().AddReference(str(manifest.asset_path))
        UsdGeom.Xformable(robot).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.85))

        # Let Kit compose the referenced USD before traversing.
        for _ in range(10):
            simulation_app.update()

        joint_names: list[str] = []
        for prim in Usd.PrimRange(robot):
            type_name = prim.GetTypeName()
            if "Joint" in type_name or prim.HasAPI("PhysicsJointAPI"):
                joint_names.append(prim.GetName())
        supported = select_supported_g1_joints(joint_names)
        if not supported:
            raise RuntimeError(
                "No supported G1 hip/knee/ankle/shoulder/elbow/waist joints were discovered. "
                f"Loaded asset: {manifest.asset_path}. Check that this is a full articulated G1 USD, not a mesh-only file."
            )

        segments = compile_humanoid_motion(load_events(args.trace))
        frames = [build_g1_policy_frame(segment, supported) for segment in segments]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "asset": str(manifest.asset_path),
                    "joint_count": len(joint_names),
                    "supported_groups": supported,
                    "frames": [
                        {"frame": frame.frame, "gesture": frame.gesture, "positions": frame.positions}
                        for frame in frames
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[g1-policy] discovered {len(joint_names)} joints")
        print(f"[g1-policy] wrote {args.output}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
