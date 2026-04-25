from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.isaac_assets import (  # noqa: E402
    UNITREE_MODEL_REPO,
    UNITREE_ROS_REPO,
    UnitreeG1AssetManifest,
    build_unitree_asset_fetch_plan,
    discover_or_raise_unitree_g1_asset,
    write_unitree_g1_manifest,
)


DEFAULT_TARGET = ROOT / "artifacts" / "isaac" / "vendor" / "unitree"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("[unitree-assets]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _clone_if_missing(repo: str, destination: Path) -> None:
    if destination.exists():
        print(f"[unitree-assets] already present: {destination}", flush=True)
    else:
        _run(["git", "clone", repo, str(destination)])
    if shutil.which("git-lfs") is not None:
        _run(["git", "lfs", "pull"], cwd=destination)


def fetch_assets(target_dir: Path, *, dry_run: bool = False) -> Path:
    plan = build_unitree_asset_fetch_plan(target_dir)
    print(f"[unitree-assets] target: {plan.target_dir}", flush=True)
    if dry_run:
        for command in plan.commands:
            print(command)
        return plan.manifest_path

    if shutil.which("git") is None:
        raise RuntimeError("git is required. Install it on the Brev host before fetching Unitree assets.")
    if shutil.which("git-lfs") is None:
        print(
            "[unitree-assets] WARNING: git-lfs not found. If Hugging Face files are LFS pointers, run: "
            "sudo apt-get update && sudo apt-get install -y git-lfs",
            flush=True,
        )
    else:
        _run(["git", "lfs", "install"])

    try:
        plan.target_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        isaac_art = ROOT / "artifacts" / "isaac"
        raise RuntimeError(
            f"Permission denied creating {plan.target_dir}. "
            "This often happens when Docker/Isaac created `artifacts/isaac` as root. "
            "On the Brev host run:\n"
            f"  sudo chown -R \"$USER:$USER\" {isaac_art}\n"
            "or remove the vendor tree and retry:\n"
            f"  sudo rm -rf {isaac_art / 'vendor'}\n"
            "Then re-run this script."
        ) from exc
    _clone_if_missing(UNITREE_MODEL_REPO, plan.target_dir / "unitree_model")
    _clone_if_missing(UNITREE_ROS_REPO, plan.target_dir / "unitree_ros")

    asset = discover_or_raise_unitree_g1_asset(plan.target_dir)
    manifest = UnitreeG1AssetManifest.from_asset(
        root=plan.target_dir,
        asset_path=asset,
        source="unitree_model" if "unitree_model" in asset.parts else "unitree_ros",
    )
    path = write_unitree_g1_manifest(manifest)
    print(f"[unitree-assets] wrote manifest: {path}", flush=True)
    print(f"[unitree-assets] G1 asset: {asset}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official Unitree G1 assets for the Isaac replay.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fetch_assets(args.target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
