import importlib.util
from pathlib import Path


def _load_view_script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "mujoco" / "view_dynamic_grasp.py"
    spec = importlib.util.spec_from_file_location("view_dynamic_grasp", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_is_mujoco_task_policy_noise_line_matches_spam() -> None:
    mod = _load_view_script()
    s = "2026-01-01 mjpython[1:2] Task policy set failed: 4 ((os/kern) invalid argument)"
    assert mod._is_mujoco_task_policy_noise_line(s)


def test_is_mujoco_task_policy_noise_line_keeps_normal_lines() -> None:
    mod = _load_view_script()
    assert not mod._is_mujoco_task_policy_noise_line("keep this traceback or warning\n")
    assert not mod._is_mujoco_task_policy_noise_line("Task policy set failed: something else")
