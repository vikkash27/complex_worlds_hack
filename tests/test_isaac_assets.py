from pathlib import Path

import pytest

from robocerebra_rl.isaac_assets import (
    UNITREE_G1_URDF_RELATIVE_PATHS,
    UnitreeG1AssetManifest,
    build_unitree_asset_fetch_plan,
    find_unitree_g1_asset,
    load_unitree_g1_manifest,
    write_unitree_g1_manifest,
)


def test_build_unitree_asset_fetch_plan_targets_official_sources(tmp_path):
    target = tmp_path / "unitree"

    plan = build_unitree_asset_fetch_plan(target)

    assert plan.target_dir == target
    assert "https://huggingface.co/datasets/unitreerobotics/unitree_model" in plan.commands[0]
    assert "https://github.com/unitreerobotics/unitree_ros.git" in plan.commands[1]
    assert plan.manifest_path == target / "unitree_g1_manifest.json"


def test_find_unitree_g1_asset_prefers_usd_then_urdf(tmp_path):
    usd = tmp_path / "unitree_model" / "G1" / "g1.usd"
    urdf = tmp_path / "unitree_ros" / "robots" / "g1_description" / "g1_29dof.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot name='g1' />", encoding="utf-8")

    assert find_unitree_g1_asset(tmp_path) == urdf

    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")

    assert find_unitree_g1_asset(tmp_path) == usd
    assert any("g1_29dof.urdf" in path for path in UNITREE_G1_URDF_RELATIVE_PATHS)


def test_manifest_round_trips_with_existing_asset(tmp_path):
    asset = tmp_path / "unitree_model" / "G1" / "g1.usd"
    asset.parent.mkdir(parents=True)
    asset.write_text("#usda 1.0", encoding="utf-8")
    manifest = UnitreeG1AssetManifest.from_asset(root=tmp_path, asset_path=asset, source="unitree_model")

    path = write_unitree_g1_manifest(manifest)

    loaded = load_unitree_g1_manifest(path)
    assert loaded.asset_path == asset
    assert loaded.asset_kind == "usd"
    assert loaded.source == "unitree_model"


def test_manifest_rejects_missing_asset(tmp_path):
    manifest = UnitreeG1AssetManifest(
        root=tmp_path,
        asset_path=tmp_path / "missing.usd",
        asset_kind="usd",
        source="unitree_model",
    )

    with pytest.raises(FileNotFoundError, match="Unitree G1 asset not found"):
        manifest.require_existing_asset()


def test_manifest_rejects_git_lfs_pointer_asset(tmp_path):
    asset = tmp_path / "unitree_model" / "G1" / "g1.usd"
    asset.parent.mkdir(parents=True)
    asset.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc123\n"
        "size 123456\n",
        encoding="utf-8",
    )
    manifest = UnitreeG1AssetManifest.from_asset(root=tmp_path, asset_path=asset, source="unitree_model")

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        manifest.require_existing_asset()
