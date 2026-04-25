from __future__ import annotations

import math

import numpy as np

from robocerebra_rl.humanoid_motion import MotionSegment

# World-frame props in front of the G1 (pelvis ~0,0,0.793); Y forward, Z up.
_STATION_BENCH: dict[str, tuple[float, float, float]] = {
    "pantry": (-0.48, 0.46, 0.40),
    "counter": (-0.22, 0.46, 0.90),
    "sink": (0.02, 0.46, 0.88),
    "table": (0.28, 0.46, 0.90),
    "delivery": (0.52, 0.46, 0.88),
}

MOCAP_BODY_NAMES = (
    "rc_mocap_spill",
    "rc_mocap_tote",
)


def _bench_for_station(name: str) -> tuple[float, float, float]:
    return _STATION_BENCH.get(name, _STATION_BENCH["counter"])


def artifact_positions_for_segment(
    segment: MotionSegment,
    *,
    frame_index: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """World positions (x, y, z) for each named mocap body for the given segment."""
    st = str(segment.station) if segment.station else "counter"
    phase = str(segment.phase or "navigate")
    g = str(segment.gesture)
    x0, y0, z0 = _bench_for_station(st)

    wobble = 0.0
    if g == "stalled" or segment.status == "failed":
        wobble = 0.025 * math.sin(frame_index * 0.35)
    reach = 0.0
    if g in {"grasp", "place", "recover", "handoff"} or segment.status == "recovery":
        reach = 0.1
    if g == "stalled":
        reach = 0.0

    ya = y0 - reach
    zt = z0 + 0.02 + wobble
    tray = (x0 + wobble, ya, zt)
    mug = (x0 + 0.08 + wobble, ya + 0.04, zt + 0.08)

    # Spill puddle: visible at sink (and during recovery) so it reads as "mop-up"
    sp_z = 0.87 + (0.02 if g == "recover" else 0.0) + 0.01 * math.sin(frame_index * 0.2)
    sx, sy, _ = _bench_for_station("sink")
    show_spill = st == "sink" or "sink" in phase or (segment.status == "recovery" and g == "recover")
    spill = (sx, sy, sp_z) if show_spill else (sx, -8.0, 0.0)

    # Snack / pantry tote: fixed at pantry so the lab layout stays readable
    px, py, pz = _bench_for_station("pantry")
    tote = (px, py, pz)

    return {
        "tray": tray,
        "mug_ref": mug,
        "spill": spill,
        "tote": tote,
    }


def default_artifact_home_positions() -> dict[str, tuple[float, float, float]]:
    """Rest layout when no segment: props spread on the workbench."""
    b = _bench_for_station("counter")
    return {
        "tray": b,
        "mug_ref": (b[0] + 0.08, b[1] + 0.04, 0.98),
        "spill": (*_bench_for_station("sink")[:2], 0.87),
        "tote": _bench_for_station("pantry"),
    }


def apply_artifact_mocap(mujoco, model, data, segment: MotionSegment | None, *, frame_index: int) -> None:
    """Set mocap body poses from RoboCerebra / hospitality task context."""
    if segment is not None:
        pos_map = artifact_positions_for_segment(segment, frame_index=frame_index)
    else:
        pos_map = default_artifact_home_positions()
    quat = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    mocap_from_key = {
        "rc_mocap_spill": "spill",
        "rc_mocap_tote": "tote",
    }
    for mname, k in mocap_from_key.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mname)
        if bid < 0:
            continue
        mid = int(model.body_mocapid[bid])
        if mid < 0:
            continue
        p = pos_map.get(k, (0.0, -10.0, 0.0))
        data.mocap_pos[mid, 0] = float(p[0])
        data.mocap_pos[mid, 1] = float(p[1])
        data.mocap_pos[mid, 2] = float(p[2])
        data.mocap_quat[mid, :] = quat
