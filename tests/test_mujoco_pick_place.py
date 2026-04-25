from pathlib import Path

import numpy as np
import pytest

from robocerebra_rl.humanoid_motion import MotionSegment
from robocerebra_rl.mujoco_pick_place import (
    WELD_REL_P,
    apply_task_tray_pick_place,
)


def _seg(gesture: str, **kwargs: object) -> MotionSegment:
    return MotionSegment(
        frame=1,
        action="x",
        station="counter",
        phase="navigate",
        gesture=gesture,
        root_xyz=(0.0, 0.0, 0.0),
        left_hand_xyz=(0.0, 0.0, 0.0),
        right_hand_xyz=(0.0, 0.0, 0.0),
        caption="",
        status=str(kwargs.get("status", "success")),
        progress_fraction=0.0,
        tool_call_index=1,
    )


def test_weld_rel_matches_mjcf_constant() -> None:
    assert WELD_REL_P.shape == (3,)


def test_apply_task_tray_sets_free_joint_bench_on_walk():
    pytest.importorskip("mujoco")
    import mujoco
    from robocerebra_rl.mujoco_assets import find_menagerie_g1_scene, write_task_showcase_scene

    root = Path(__file__).resolve().parents[1] / "artifacts" / "mujoco" / "vendor"
    scene = find_menagerie_g1_scene(root)
    if scene is None:
        pytest.skip("menagerie G1 not checked out")
    xml = write_task_showcase_scene(scene.parent)
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    seg = _seg("walk")
    apply_task_tray_pick_place(mujoco, m, d, seg, frame_index=0)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rc_tray_free")
    assert jid >= 0
    adr = m.jnt_qposadr[jid]
    p = d.qpos[adr : adr + 3]
    assert float(np.linalg.norm(p)) > 0.1


def test_apply_task_tray_held_uses_wrist_fk_on_grasp():
    pytest.importorskip("mujoco")
    import mujoco
    from robocerebra_rl.mujoco_assets import find_menagerie_g1_scene, write_task_showcase_scene
    from robocerebra_rl.mujoco_g1_policy import build_mujoco_g1_qpos, default_mujoco_g1_joint_names

    root = Path(__file__).resolve().parents[1] / "artifacts" / "mujoco" / "vendor"
    scene = find_menagerie_g1_scene(root)
    if scene is None:
        pytest.skip("menagerie G1 not checked out")
    xml = write_task_showcase_scene(scene.parent)
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    seg = _seg("grasp")
    q = build_mujoco_g1_qpos(seg, default_mujoco_g1_joint_names())
    t = dict(zip(default_mujoco_g1_joint_names(), q, strict=True))
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n and n in t:
            d.qpos[m.jnt_qposadr[j]] = t[n]
    mujoco.mj_forward(m, d)
    apply_task_tray_pick_place(mujoco, m, d, seg, frame_index=0)
    mujoco.mj_forward(m, d)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rc_tray_free")
    adr = m.jnt_qposadr[jid]
    p_new = d.qpos[adr : adr + 3].copy()
    assert p_new[2] > 0.2
