from __future__ import annotations

"""Runtime dynamics utilities: weld relpose from measured body poses, PD targets, and base anchoring."""

import numpy as np

from robocerebra_rl.mujoco_pick_place import _should_carry_tray

WELD_EQ_NAME = "rc_weld_tray_palm"
WRIST_BODY = "right_wrist_yaw_link"
TRAY_BODY = "rc_task_tray"


def body_T_world(mujoco, d, body_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    p = d.xpos[body_id]
    T[0:3, 3] = p
    Rm = np.zeros(9, dtype=np.float64)
    q = np.array(d.xquat[body_id], dtype=np.float64)
    mujoco.mju_quat2Mat(Rm, q)
    T[0:3, 0:3] = Rm.reshape(3, 3, order="C")
    return T


def measured_weld_eq7_body2_in_body1(mujoco, d, body1_id: int, body2_id: int) -> np.ndarray:
    """Return 7 values for ``m.eq_data[eid, 3:10]``: (px,py,pz, qw,qx,qy,qz) — pose of *body2* in *body1* frame.

    This matches the MuJoCo packaged equality row layout for a body–body ``weld`` (after the leading three zeros).
    """
    t1 = body_T_world(mujoco, d, body1_id)
    t2 = body_T_world(mujoco, d, body2_id)
    t1_inv = np.linalg.inv(t1)
    t_rel = t1_inv @ t2
    p = t_rel[0:3, 3]
    r = t_rel[0:3, 0:3]
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, r.reshape(-1, order="C"))
    return np.concatenate([p, quat], axis=0)


def apply_weld_relpose_to_model(
    model,
    eid: int,
    eq7: np.ndarray,
) -> None:
    """Write measured relpose (7) into the weld’s ``eq_data`` row (columns 3:10)."""
    if eid < 0 or eid >= model.neq or eq7.shape[0] != 7:
        return
    model.eq_data[eid, 3:10] = eq7


def set_equality_on(data, eid: int, on: bool) -> None:
    if 0 <= eid < int(data.eq_active.shape[0]):
        data.eq_active[eid] = 1 if on else 0


def clear_tray_velocities(m, d, mujoco) -> None:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rc_tray_free")
    if j < 0:
        return
    da = int(m.jnt_dofadr[j])
    d.qvel[da : da + 6] = 0.0


def anchor_floating_base(m, d, key_q7: np.ndarray) -> None:
    """Pin the 7-DoF floating base (pos + quat) to avoid drift on the 6 base velocities."""
    d.qpos[0:7] = key_q7
    d.qvel[0:6] = 0.0


def apply_position_actuator_targets(
    m,
    d,
    mujoco,
    joint_name_to_target: dict[str, float],
) -> None:
    """G1 :position actuators are named like joints; set ``ctrl`` to the desired joint position."""
    for jname, target in joint_name_to_target.items():
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid >= 0 and aid < m.nu:
            d.ctrl[aid] = float(target)


def should_toggle_grasp(prev: MotionSegment | None, cur: MotionSegment | None) -> tuple[bool, bool]:
    """(activate_weld, release_weld) for one logical segment step."""
    if cur is None:
        return (False, False)
    c = _should_carry_tray(cur)
    p = _should_carry_tray(prev) if prev is not None else False
    return (c and not p, p and not c)
