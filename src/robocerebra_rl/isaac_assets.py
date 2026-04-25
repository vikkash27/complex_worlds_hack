from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


UNITREE_MODEL_REPO = "https://huggingface.co/datasets/unitreerobotics/unitree_model"
UNITREE_ROS_REPO = "https://github.com/unitreerobotics/unitree_ros.git"

UNITREE_G1_USD_RELATIVE_PATHS = (
    "unitree_model/G1/g1.usd",
    "unitree_model/g1/g1.usd",
    "unitree_model/robots/g1/g1.usd",
    "unitree_model/robots/G1/g1.usd",
)
UNITREE_G1_URDF_RELATIVE_PATHS = (
    "unitree_ros/robots/g1_description/g1_29dof.urdf",
    "unitree_ros/unitree_ros/robots/g1_description/g1_29dof.urdf",
    "unitree_ros/g1_description/urdf/g1_29dof.urdf",
    "unitree_ros/robots/g1_description/urdf/g1_29dof.urdf",
)


@dataclass(frozen=True)
class UnitreeAssetFetchPlan:
    target_dir: Path
    manifest_path: Path
    commands: tuple[str, ...]


@dataclass(frozen=True)
class UnitreeG1AssetManifest:
    root: Path
    asset_path: Path
    asset_kind: str
    source: str

    @property
    def manifest_path(self) -> Path:
        return self.root / "unitree_g1_manifest.json"

    @classmethod
    def from_asset(cls, *, root: Path, asset_path: Path, source: str) -> "UnitreeG1AssetManifest":
        suffix = asset_path.suffix.lower().lstrip(".")
        asset_kind = "usd" if suffix in {"usd", "usda", "usdc"} else suffix
        return cls(root=root, asset_path=asset_path, asset_kind=asset_kind, source=source)

    def require_existing_asset(self) -> Path:
        if not self.asset_path.is_file():
            raise FileNotFoundError(
                f"Unitree G1 asset not found at {self.asset_path}. "
                "Run scripts/isaac/fetch_unitree_g1_assets.py on the Brev host first."
            )
        prefix = self.asset_path.read_bytes()[:128]
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"Unitree G1 asset at {self.asset_path} is a Git LFS pointer, not the real model. "
                "Install git-lfs and run `git lfs pull` in the fetched Unitree asset repository."
            )
        return self.asset_path

    def as_json(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "asset_path": str(self.asset_path),
            "asset_kind": self.asset_kind,
            "source": self.source,
        }


def build_unitree_asset_fetch_plan(target_dir: Path) -> UnitreeAssetFetchPlan:
    target = target_dir.expanduser()
    return UnitreeAssetFetchPlan(
        target_dir=target,
        manifest_path=target / "unitree_g1_manifest.json",
        commands=(
            f"git clone {UNITREE_MODEL_REPO} {target / 'unitree_model'}",
            f"git clone {UNITREE_ROS_REPO} {target / 'unitree_ros'}",
        ),
    )


def find_unitree_g1_asset(root: Path) -> Path | None:
    base = root.expanduser()
    for relative in (*UNITREE_G1_USD_RELATIVE_PATHS, *UNITREE_G1_URDF_RELATIVE_PATHS):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    for pattern in ("**/g1.usd", "**/g1_minimal.usd", "**/g1_29dof.urdf", "**/g1_23dof.urdf"):
        for candidate in sorted(base.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def discover_or_raise_unitree_g1_asset(root: Path) -> Path:
    asset = find_unitree_g1_asset(root)
    if asset is None:
        raise FileNotFoundError(
            f"No Unitree G1 USD/URDF found under {root}. "
            "Run `python scripts/isaac/fetch_unitree_g1_assets.py` on the Brev host."
        )
    return asset


def write_unitree_g1_manifest(manifest: UnitreeG1AssetManifest) -> Path:
    manifest.root.mkdir(parents=True, exist_ok=True)
    path = manifest.manifest_path
    path.write_text(json.dumps(manifest.as_json(), indent=2), encoding="utf-8")
    return path


def load_unitree_g1_manifest(path: Path) -> UnitreeG1AssetManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = UnitreeG1AssetManifest(
        root=Path(payload["root"]),
        asset_path=Path(payload["asset_path"]),
        asset_kind=str(payload["asset_kind"]),
        source=str(payload["source"]),
    )
    manifest.require_existing_asset()
    return manifest
