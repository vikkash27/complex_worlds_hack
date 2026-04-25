import numpy as np

import pytest

from robocerebra_rl.mujoco_dynamic import measured_weld_eq7_body2_in_body1


def test_measured_weld_identity_without_mujoco_se3():
    """Same two transforms → relative should be (near) zero translation, identity quat."""
    t1 = np.eye(4)
    t2 = np.eye(4)
    t1_inv = np.linalg.inv(t1)
    t_rel = t1_inv @ t2
    p = t_rel[0:3, 3]
    r = t_rel[0:3, 0:3]
    assert np.allclose(p, 0.0, atol=1e-6)
    assert np.allclose(r, np.eye(3), atol=1e-6)


def test_measured_weld_matches_mujoco_body_helper():
    pytest.importorskip("mujoco")
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="a" pos="0 0 0"><joint type="free" name="ja"/><geom type="sphere" size="0.05" mass="0.1"/></body>
        <body name="b" pos="0.1 0.02 0.03" quat="1 0 0 0"><joint type="free" name="jb"/><geom type="box" size="0.04 0.04 0.04" mass="0.1"/></body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    ba = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a")
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")
    eq7 = measured_weld_eq7_body2_in_body1(mujoco, d, ba, bb)
    assert eq7.shape == (7,)
    assert float(eq7[0]) > 0.05
    assert float(eq7[3]) > 0.9
