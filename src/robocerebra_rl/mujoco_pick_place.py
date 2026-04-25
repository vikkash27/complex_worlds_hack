from __future__ import annotations

"""Kinematic pick–place for the G1 task tray, consistent with the MJCF :equality/:weld relpose.

The scene defines ``rc_weld_tray_palm`` (inactive by default). Each frame we set ``rc_tray_free``
qpos so the tray (and attached mug geoms) matches ``T_world_wrist @ T_relpose`` for carry phases,
or the bench position otherwise — the same relationship an active weld enforces, without
inverting dynamics for offline video.
"""

import math

import numpy as np

from robocerebra_rl.humanoid_motion import MotionSegment

# Must match the weld ``relpose="px py pz qx qy qz qw"`` in :data:`_TASK_SHOWCASE_MJCF` in mujoco_assets.
# MuJoCo stores weld relpose as: 3D position, then quaternion w x y z.
WELD_REL_P = np.array((0.14, 0.02, -0.04), dtype=np.float64)
WELD_REL_QUAT = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)

WRIST_BODY_NAME = "right_wrist_yaw_link"
TRAY_JOINT_NAME = "rc_tray_free"

_T_WELD_4: np.ndarray | None = None


def _T_from_p_R(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def _build_T_weld() -> np.ndarray:
    muj = __import__("mujoco", fromlist=["mju_quat2Mat"])
    T = np.eye(4, dtype=np.float64)
    T[0:3, 3] = WELD_REL_P
    Rm = np.zeros(9, dtype=np.float64)
    muj.mju_quat2Mat(Rm, WELD_REL_QUAT)
    T[0:3, 0:3] = Rm.reshape(3, 3, order="C")
    return T


def _T_weld_4x4() -> np.ndarray:
    global _T_WELD_4
    if _T_WELD_4 is None:
        _T_WELD_4 = _build_T_weld()
    return _T_WELD_4


def _should_carry_tray(segment: MotionSegment) -> bool:
    if segment.gesture in {"stalled"} or segment.status == "failed":
        return False
    return str(segment.gesture) in {"grasp", "place", "handoff", "recover"}


def set_tray_free_joint(
    mujoco,
    model,
    data,
    *,
    pos: np.ndarray,
    quat: np.ndarray,
) -> None:
    """Set ``rc_tray_free`` 7-DoF pose (x, y, z) + (w, x, y, z)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TRAY_JOINT_NAME)
    if jid < 0:
        return
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat


def _pose_tray_held(mujoco, model, data, *, wrist_bid: int) -> None:
    p1 = np.array(data.xpos[wrist_bid], copy=True, dtype=np.float64)
    q1 = np.array(data.xquat[wrist_bid], copy=True, dtype=np.float64)
    R1m = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R1m, q1)
    R1 = R1m.reshape(3, 3, order="C")
    T1 = _T_from_p_R(p1, R1)
    T2 = T1 @ _T_weld_4x4()
    p2 = T2[0:3, 3]
    R2 = T2[0:3, 0:3]
    quat2 = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat2, R2.reshape(-1, order="C"))
    set_tray_free_joint(mujoco, model, data, pos=p2, quat=quat2)


def apply_task_tray_pick_place(
    mujoco,
    model,
    data,
    segment: MotionSegment | None,
    *,
    frame_index: int,
) -> None:
    from robocerebra_rl.mujoco_artifact_layout import (
        artifact_positions_for_segment,
        default_artifact_home_positions,
    )

    w_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY_NAME)
    if w_bid < 0:
        return
    if segment is None:
        pos = default_artifact_home_positions()["tray"]
        set_tray_free_joint(
            mujoco, model, data, pos=np.asarray(pos, dtype=np.float64), quat=np.array((1, 0, 0, 0), float)
        )
        return
    if not _should_carry_tray(segment):
        t = artifact_positions_for_segment(segment, frame_index=frame_index)
        p = t["tray"]
        wobble = 0.0
        if segment.gesture == "stalled" or segment.status == "failed":
            wobble = 0.02 * math.sin(frame_index * 0.4)
        set_tray_free_joint(
            mujoco,
            model,
            data,
            pos=np.array((p[0] + wobble, p[1], p[2]), dtype=np.float64),
            quat=np.array((1, 0, 0, 0), float),
        )
        return
    _pose_tray_held(mujoco, model, data, wrist_bid=w_bid)
