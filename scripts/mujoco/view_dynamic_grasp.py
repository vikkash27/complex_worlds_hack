#!/usr/bin/env python3
"""Interactive MuJoCo (Simulate) viewer: PD-driven G1, measured weld, mj_step.

This is the **live 3D “demo renderer”** for the same RoboCerebra / hospitality trace as
``render_g1_showcase.py``; it is **not** a GIF exporter.

The floating base is re-anchored to the `stand` keyframe each substep (position + vels) so
the humanoid does not float away while the arm tracks the trace; the free tray is driven by
physics + equality when grasp is **on** (or kinematic pre-align, then a measured weld relpose).

Run (from repo root, venv on):

  .venv/bin/python scripts/mujoco/view_dynamic_grasp.py --trace artifacts/traces/humanoid_trained_long_horizon.jsonl

On **macOS**, ``mujoco.viewer.launch_passive`` must run under ``mjpython`` (not plain
``python``). This script **re-exec**s itself via ``.venv/bin/mjpython`` when needed. You can
also invoke ``.venv/bin/mjpython`` directly.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robocerebra_rl.mujoco_assets import load_menagerie_g1_manifest
from robocerebra_rl.mujoco_dynamic import (
    WELD_EQ_NAME,
    anchor_floating_base,
    apply_position_actuator_targets,
    apply_weld_relpose_to_model,
    clear_tray_velocities,
    measured_weld_eq7_body2_in_body1,
    set_equality_on,
    should_toggle_grasp,
)
from robocerebra_rl.mujoco_g1_policy import build_mujoco_g1_qpos, default_mujoco_g1_joint_names
from robocerebra_rl.mujoco_pick_place import WRIST_BODY_NAME, apply_task_tray_pick_place
from robocerebra_rl.mujoco_showcase import _active_segment, build_showcase_timeline, load_trace


def _set_named_qpos(m, d, mujoco, segment) -> None:
    n = default_mujoco_g1_joint_names()
    qpos = build_mujoco_g1_qpos(segment, n)
    targets = dict(zip(n, qpos, strict=True))
    for joint_index in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
        if not name or name not in targets:
            continue
        d.qpos[m.jnt_qposadr[joint_index]] = targets[name]


def _is_mujoco_task_policy_noise_line(line: str) -> bool:
    """True for macOS ``mjpython``/Cocoa ``Task policy set failed`` noise (harmless)."""
    return "Task policy set failed" in line and "invalid argument" in line


_fd2_mujoco_stderr_filter_installed: bool = False


def _stderr_filter_reader_loop(read_fd: int, term_fd: int) -> None:
    buf = b""
    try:
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line_bytes = buf[: nl + 1]
                buf = buf[nl + 1 :]
                body = line_bytes[:-1].decode("utf-8", errors="replace").rstrip("\r")
                if not _is_mujoco_task_policy_noise_line(body):
                    os.write(term_fd, line_bytes)
        if buf:
            os.write(term_fd, buf)
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


def _maybe_install_macos_mujoco_stderr_filter(*, enabled: bool) -> None:
    """Filter known ``Task policy set failed`` spam on **fd 2**, not just ``sys.stderr``.

    MuJoco/Qt/Cocoa and macOS system frameworks often write to **C stderr** (file descriptor 2)
    without going through Python's ``sys.stderr`` object, so wrapping ``sys.stderr`` alone does
    nothing.  We interpose a pipe on fd 2 and forward filtered bytes to a duplicate of the
    original TTY.
    """
    global _fd2_mujoco_stderr_filter_installed
    if not enabled or sys.platform != "darwin" or _fd2_mujoco_stderr_filter_installed:
        return
    try:
        term_fd = os.dup(2)
    except OSError:
        return
    try:
        r_fd, w_fd = os.pipe()
    except OSError:
        try:
            os.close(term_fd)
        except OSError:
            pass
        return
    try:
        os.dup2(w_fd, 2)
    except OSError:
        try:
            os.close(term_fd)
        except OSError:
            pass
        try:
            os.close(r_fd)
        except OSError:
            pass
        try:
            os.close(w_fd)
        except OSError:
            pass
        return
    try:
        os.close(w_fd)
    except OSError:
        pass
    thr = threading.Thread(
        target=_stderr_filter_reader_loop,
        args=(r_fd, term_fd),
        name="mujoco-stderr-fd2-filter",
        daemon=True,
    )
    try:
        thr.start()
    except Exception:
        try:
            os.dup2(term_fd, 2)
        except OSError:
            pass
        try:
            os.close(term_fd)
        except OSError:
            pass
        try:
            os.close(r_fd)
        except OSError:
            pass
        return
    # All future writes to fd 2 (Python + native) go to the pipe; the thread forwards to term_fd.
    # Use FileIO(2) + TextIOWrapper — ``open(2, ...)`` can hang on some macOS/Python builds after ``dup2`` to a pipe.
    try:
        sys.stderr = io.TextIOWrapper(
            io.FileIO(2, "w", closefd=False),
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        sys.__stderr__ = sys.stderr
    except OSError:
        try:
            os.dup2(term_fd, 2)
        except OSError:
            pass
        try:
            os.close(term_fd)
        except OSError:
            pass
        return
    _fd2_mujoco_stderr_filter_installed = True


def _maybe_reexec_with_mjpython_on_macos() -> None:
    """``launch_passive`` on darwin requires mjpython; re-exec before any GUI path."""
    if sys.platform != "darwin":
        return
    if "--headless" in sys.argv:
        return
    exe = Path(sys.executable)
    if "mjpython" in exe.name.lower():
        return
    mj = exe.parent / "mjpython"
    if not mj.is_file():
        w = shutil.which("mjpython")
        mj = Path(w) if w else mj
    if not mj.is_file():
        raise SystemExit(
            "On macOS the MuJoCo viewer needs `mjpython` (install `mujoco` in this venv). "
            "Expected at .venv/bin/mjpython — not found next to " + str(exe)
        )
    script = Path(__file__).resolve()
    os.execv(str(mj), [str(mj), str(script), *sys.argv[1:]])


def main() -> None:
    _maybe_reexec_with_mjpython_on_macos()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "mujoco" / "vendor" / "menagerie_g1_manifest.json")
    p.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "artifacts" / "traces" / "humanoid_trained_long_horizon.jsonl",
        help="OpenReward / humanoid JSONL; drives the motion segment timeline.",
    )
    p.add_argument("--lane", choices=("optimized", "baseline"), default="optimized", help="Which policy lane to play.")
    p.add_argument("--keyframe", type=int, default=0, help="Key index for the standing init (0 = 'stand' if present).")
    p.add_argument(
        "--frame-step-seconds",
        type=float,
        default=0.45,
        help="Wall-clock seconds per logical `MotionSegment` frame in the trace.",
    )
    p.add_argument("--substeps", type=int, default=2, help="mj_step calls per sync (× viewer rate).")
    p.add_argument("--headless", action="store_true", help="No GUI: run a short physics test and exit (for CI or SSH).")
    p.add_argument(
        "--no-filter-macos-stderr",
        action="store_true",
        help="On macOS, do not suppress known harmless 'Task policy set failed' lines from mjpython/Cocoa (verbose).",
    )
    args = p.parse_args()

    _maybe_install_macos_mujoco_stderr_filter(enabled=(sys.platform == "darwin" and not args.headless and not args.no_filter_macos_stderr))

    import mujoco
    from mujoco import viewer as mj_viewer

    scene = load_menagerie_g1_manifest(args.manifest).ensure_task_showcase_scene()
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    events = load_trace(args.trace)
    timeline = build_showcase_timeline(events, events)
    segs = list(timeline.optimized if args.lane == "optimized" else timeline.baseline)
    max_f = int(timeline.max_frame)

    wrist_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY_NAME)
    tray_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rc_task_tray")
    eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_EQ_NAME)
    if wrist_b < 0 or tray_b < 0 or eid < 0:
        raise SystemExit("Missing bodies or weld in scene; re-run ensure_task_showcase_scene()")

    if m.nkey < 1:
        raise SystemExit("No keyframes; expected g1 'stand' key in scene.")
    mujoco.mj_resetDataKeyframe(m, d, int(args.keyframe))
    key_q7 = d.qpos[0:7].copy()

    seg0 = segs[0] if segs else None
    if seg0 is not None:
        _set_named_qpos(m, d, mujoco, seg0)
    mujoco.mj_forward(m, d)
    apply_task_tray_pick_place(mujoco, m, d, seg0, frame_index=0)
    from robocerebra_rl.mujoco_artifact_layout import apply_artifact_mocap  # local

    apply_artifact_mocap(mujoco, m, d, seg0, frame_index=0)
    mujoco.mj_forward(m, d)
    d.eq_active[eid] = 0
    carrying: bool = False
    prev: object = None

    names = default_mujoco_g1_joint_names()
    n_steps = 0
    substep = max(1, int(args.substeps))
    fstep = max(0.05, float(args.frame_step_seconds))

    def one_physics_block(cur_seg, fidx: int) -> None:
        nonlocal carrying, prev
        tdict: dict[str, float] = {}
        if cur_seg is not None:
            qv = build_mujoco_g1_qpos(cur_seg, names)
            tdict = dict(zip(names, qv, strict=True))
        apply_position_actuator_targets(m, d, mujoco, tdict)
        act, rel = should_toggle_grasp(prev, cur_seg)
        if act and cur_seg is not None:
            apply_task_tray_pick_place(mujoco, m, d, cur_seg, frame_index=fidx)
            mujoco.mj_forward(m, d)
            eq7 = measured_weld_eq7_body2_in_body1(mujoco, d, wrist_b, tray_b)
            apply_weld_relpose_to_model(m, eid, eq7)
            clear_tray_velocities(m, d, mujoco)
            set_equality_on(d, eid, True)
            carrying = True
        if rel and carrying:
            set_equality_on(d, eid, False)
            carrying = False
            clear_tray_velocities(m, d, mujoco)
        prev = cur_seg
        apply_artifact_mocap(mujoco, m, d, cur_seg, frame_index=fidx)
        for _ in range(substep):
            mujoco.mj_step(m, d)
            anchor_floating_base(m, d, key_q7)

    if args.headless:
        for h in range(120):
            ft = h % (max_f + 1) if max_f > 0 else 0
            cur = _active_segment(segs, ft)
            one_physics_block(cur, ft)
        print("[view_dynamic_grasp] headless smoke: 120 substeps ok")
        return

    with mj_viewer.launch_passive(m, d) as v:
        t_wall = time.time()
        while v.is_running():
            elapsed = time.time() - t_wall
            if max_f > 0:
                frame_t = int(elapsed // fstep) % (max_f + 1)
            else:
                frame_t = 0
            cur = _active_segment(segs, frame_t)
            one_physics_block(cur, frame_t)
            mujoco.mj_forward(m, d)
            v.sync()
            n_steps += 1
            if n_steps > 1_000_000:
                break


if __name__ == "__main__":
    main()
