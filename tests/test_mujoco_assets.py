from pathlib import Path

import pytest

from robocerebra_rl.mujoco_assets import (
    MENAGERIE_REPO,
    MenagerieG1Manifest,
    build_menagerie_fetch_plan,
    find_menagerie_g1_scene,
    load_menagerie_g1_manifest,
    write_menagerie_g1_manifest,
)


def test_build_menagerie_fetch_plan_sparse_clones_unitree_g1(tmp_path):
    target = tmp_path / "menagerie"

    plan = build_menagerie_fetch_plan(target)

    assert plan.target_dir == target
    assert plan.manifest_path == target / "menagerie_g1_manifest.json"
    assert MENAGERIE_REPO in plan.commands[0]
    assert "sparse-checkout set unitree_g1" in plan.commands[1]


def test_find_menagerie_g1_scene_prefers_scene_with_hands_then_scene(tmp_path):
    root = tmp_path / "mujoco_menagerie" / "unitree_g1"
    root.mkdir(parents=True)
    scene = root / "scene.xml"
    scene.write_text("<mujoco />", encoding="utf-8")

    assert find_menagerie_g1_scene(tmp_path) == scene

    scene_with_hands = root / "scene_with_hands.xml"
    scene_with_hands.write_text("<mujoco />", encoding="utf-8")

    assert find_menagerie_g1_scene(tmp_path) == scene_with_hands


def test_menagerie_manifest_round_trips_relative_scene_path(tmp_path):
    scene = tmp_path / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
    scene.parent.mkdir(parents=True)
    scene.write_text("<mujoco />", encoding="utf-8")
    manifest = MenagerieG1Manifest.from_scene(root=tmp_path, scene_path=scene)

    path = write_menagerie_g1_manifest(manifest)

    loaded = load_menagerie_g1_manifest(path)
    assert loaded.scene_path == scene
    assert loaded.scene_relative_path == Path("mujoco_menagerie/unitree_g1/scene.xml")


def test_menagerie_manifest_rejects_missing_scene(tmp_path):
    manifest = MenagerieG1Manifest(root=tmp_path, scene_relative_path=Path("missing.xml"))

    with pytest.raises(FileNotFoundError, match="MuJoCo Menagerie Unitree G1 scene not found"):
        manifest.require_existing_scene()


def test_ensure_task_showcase_scene_writes_pick_place_wrapper(tmp_path):
    g1 = tmp_path / "unitree_g1"
    g1.mkdir(parents=True)
    (g1 / "scene_with_hands.xml").write_text(
        "<mujoco><worldbody><geom name=\"floor\" size=\"0 0 0.05\" type=\"plane\"/></worldbody></mujoco>",
        encoding="utf-8",
    )
    manifest = MenagerieG1Manifest.from_scene(root=tmp_path, scene_path=g1 / "scene_with_hands.xml")
    out = manifest.ensure_task_showcase_scene()
    assert out.suffix == ".xml"
    text = out.read_text()
    assert "rc_task_tray" in text
    assert "rc_weld_tray_palm" in text
    assert "rc_tray_free" in text


def test_task_showcase_loads_with_real_menagerie_g1():
    pytest.importorskip("mujoco")
    import mujoco

    from robocerebra_rl.mujoco_assets import find_menagerie_g1_scene, write_task_showcase_scene

    root = Path(__file__).resolve().parents[1] / "artifacts" / "mujoco" / "vendor"
    scene = find_menagerie_g1_scene(root)
    if scene is None:
        pytest.skip("menagerie G1 not checked out")
    out = write_task_showcase_scene(scene.parent)
    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.nmocap == 2
    assert m.neq == 1
