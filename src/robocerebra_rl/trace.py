from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping


@dataclass
class ToolTraceLogger:
    path: Path
    run_id: str

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id

    def record(
        self,
        *,
        tool_name: str,
        task_id: str,
        action: str | None,
        observation: Mapping[str, object],
        reward: float,
        reward_components: Mapping[str, float],
        rationale: str,
        finished: bool,
        state_hash: str,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "tool_name": tool_name,
            "task_id": task_id,
            "action": action,
            "reward": reward,
            "reward_components": dict(reward_components),
            "rationale": rationale,
            "finished": finished,
            "state_hash": state_hash,
            "observation_summary": {
                "task_name": observation.get("task_name"),
                "task_label": observation.get("task_label"),
                "ticks": observation.get("ticks"),
                "progress_fraction": observation.get("progress_fraction"),
                "expected_next": observation.get("expected_next"),
                "completed_subgoals": observation.get("completed_subgoals"),
                "last_failure_reason": observation.get("last_failure_reason"),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, sort_keys=True) + "\n")
