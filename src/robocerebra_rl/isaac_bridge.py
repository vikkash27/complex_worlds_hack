from __future__ import annotations

from dataclasses import dataclass


ISAAC_ACTION_MAP = {
    "locate_items": "Move camera/head toward tabletop and highlight mug/snack targets.",
    "clear_workspace": "Sweep distractor block away from tray path.",
    "pick_mug": "Move manipulator to mug, close gripper, lift.",
    "fill_drink": "Move mug under dispenser and animate fill level.",
    "place_snack": "Pick snack and place it onto the tray.",
    "recover_disturbance": "Counter tray bump by moving obstacle and re-centering objects.",
    "deliver_tray": "Move tray to delivery zone without dropping objects.",
}


@dataclass(frozen=True)
class SceneVariant:
    mug_position: tuple[float, float, float] = (0.35, -0.2, 0.78)
    snack_position: tuple[float, float, float] = (0.15, 0.3, 0.78)
    tray_position: tuple[float, float, float] = (0.55, 0.0, 0.78)
    disturbance_tick: int = 500
    distractor_count: int = 1
    action_failure_prob: float = 0.05
    disturbance_severity: float = 0.5

    def to_task_spec(self, seed: int) -> dict[str, object]:
        return {
            "seed": seed,
            "horizon_ticks": 1000,
            "scene": {
                "mug_position": self.mug_position,
                "snack_position": self.snack_position,
                "tray_position": self.tray_position,
                "disturbance_tick": self.disturbance_tick,
                "distractor_count": self.distractor_count,
                "action_failure_prob": self.action_failure_prob,
                "disturbance_severity": self.disturbance_severity,
            },
        }
