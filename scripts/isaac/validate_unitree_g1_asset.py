from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.isaac_assets import (  # noqa: E402
    UnitreeG1AssetManifest,
    discover_or_raise_unitree_g1_asset,
    load_unitree_g1_manifest,
    write_unitree_g1_manifest,
)


DEFAULT_TARGET = ROOT / "artifacts" / "isaac" / "vendor" / "unitree"


def validate_unitree_g1_asset(target_dir: Path = DEFAULT_TARGET) -> Path:
    manifest_path = target_dir / "unitree_g1_manifest.json"
    if manifest_path.is_file():
        manifest = load_unitree_g1_manifest(manifest_path)
        asset = manifest.require_existing_asset()
    else:
        asset = discover_or_raise_unitree_g1_asset(target_dir)
        manifest = UnitreeG1AssetManifest.from_asset(
            root=target_dir,
            asset_path=asset,
            source="unitree_model" if "unitree_model" in asset.parts else "unitree_ros",
        )
        write_unitree_g1_manifest(manifest)
    print(asset)
    return asset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local Unitree G1 asset used by Isaac replay.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    validate_unitree_g1_asset(args.target_dir)


if __name__ == "__main__":
    main()
