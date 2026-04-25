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

from robocerebra_rl.mujoco_assets import (  # noqa: E402
    MENAGERIE_REPO,
    MenagerieG1Manifest,
    build_menagerie_fetch_plan,
    find_menagerie_g1_scene,
    write_menagerie_g1_manifest,
)


DEFAULT_TARGET = ROOT / "artifacts" / "mujoco" / "vendor"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("[mujoco-assets]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def fetch_menagerie_g1(target_dir: Path, *, dry_run: bool = False) -> Path:
    plan = build_menagerie_fetch_plan(target_dir)
    if dry_run:
        for command in plan.commands:
            print(command)
        return plan.manifest_path
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch mujoco_menagerie.")
    plan.target_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = plan.target_dir / "mujoco_menagerie"
    if repo_dir.exists():
        print(f"[mujoco-assets] already present: {repo_dir}", flush=True)
    else:
        _run(["git", "clone", "--filter=blob:none", "--sparse", MENAGERIE_REPO, str(repo_dir)])
        _run(["git", "sparse-checkout", "set", "unitree_g1"], cwd=repo_dir)
    scene = find_menagerie_g1_scene(plan.target_dir)
    if scene is None:
        raise FileNotFoundError(f"Could not find Unitree G1 scene under {plan.target_dir}")
    manifest = MenagerieG1Manifest.from_scene(root=plan.target_dir, scene_path=scene)
    path = write_menagerie_g1_manifest(manifest)
    print(f"[mujoco-assets] scene: {scene}", flush=True)
    print(f"[mujoco-assets] wrote manifest: {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MuJoCo Menagerie Unitree G1 assets.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fetch_menagerie_g1(args.target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
