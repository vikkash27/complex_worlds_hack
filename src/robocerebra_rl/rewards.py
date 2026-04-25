from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, TypedDict

from robocerebra_rl.world import Transition


class GeminiScore(TypedDict):
    progress_delta: float
    subgoal_complete: bool
    irreversible_error: bool
    confidence: float
    rationale: str


def sparse_success_reward(transition: Transition) -> float:
    if not transition.done:
        return 0.0
    return 1.0 if transition.success else 0.0


def symbolic_dense_reward(transition: Transition) -> float:
    progress = transition.progress_delta * 1.5
    stage_bonus = transition.progress_after * 0.15 if transition.progress_delta > 0 else 0.0
    recovery_bonus = 0.15 if transition.disturbance_recovered and transition.progress_delta > 0 else 0.0
    mistake_penalty = -0.05 if transition.progress_delta == 0 and transition.reward < 0 else 0.0
    success_bonus = 1.0 if transition.success else 0.0
    return round(progress + stage_bonus + recovery_bonus + mistake_penalty + success_bonus, 6)


def default_symbolic_vlm_score(payload: dict[str, object]) -> GeminiScore:
    progress_delta = float(payload.get("progress_delta", 0.0))
    complete = progress_delta > 0
    return {
        "progress_delta": progress_delta,
        "subgoal_complete": complete,
        "irreversible_error": False,
        "confidence": 0.72 if complete else 0.55,
        "rationale": (
            "Cached symbolic fallback: progress is inferred from the simulator "
            "state because GEMINI_API_KEY is not configured."
        ),
    }


class GeminiRewardCache:
    def __init__(
        self,
        path: str | Path,
        scorer: Callable[[dict[str, object]], GeminiScore] | None = None,
    ) -> None:
        self.path = Path(path)
        self.scorer = scorer or default_symbolic_vlm_score
        self._cache: dict[str, GeminiScore] = self._load()

    def score(
        self,
        task_id: str,
        state_hash: str,
        subgoal: str,
        action: str,
        *,
        progress_delta: float = 0.0,
        image_path: str | None = None,
    ) -> GeminiScore:
        key = "|".join([task_id, state_hash, subgoal, action])
        if key not in self._cache:
            payload = {
                "task_id": task_id,
                "state_hash": state_hash,
                "subgoal": subgoal,
                "action": action,
                "progress_delta": progress_delta,
                "image_path": image_path,
            }
            self._cache[key] = self.scorer(payload)
            self._save()
        return self._cache[key]

    def _load(self) -> dict[str, GeminiScore]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._cache, file, indent=2, sort_keys=True)


def gemini_reward_scorer(model: str = "gemini-2.5-flash") -> Callable[[dict[str, object]], GeminiScore]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return default_symbolic_vlm_score

    from google import genai

    client = genai.Client(api_key=api_key)

    def score(payload: dict[str, object]) -> GeminiScore:
        prompt = (
            "You are scoring progress in a long-horizon robotic manipulation benchmark. "
            "Return strict JSON with progress_delta, subgoal_complete, irreversible_error, "
            "confidence, and rationale. "
            f"Task: {payload['task_id']}. Subgoal: {payload['subgoal']}. "
            f"Action: {payload['action']}. Simulator progress delta: {payload['progress_delta']}."
        )
        response = client.models.generate_content(model=model, contents=prompt)
        text = getattr(response, "text", "") or "{}"
        try:
            parsed = json.loads(text.strip().strip("`"))
        except json.JSONDecodeError:
            parsed = default_symbolic_vlm_score(payload)
            parsed["rationale"] = text[:300] or parsed["rationale"]
        return {
            "progress_delta": float(parsed.get("progress_delta", payload.get("progress_delta", 0.0))),
            "subgoal_complete": bool(parsed.get("subgoal_complete", False)),
            "irreversible_error": bool(parsed.get("irreversible_error", False)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "rationale": str(parsed.get("rationale", ""))[:500],
        }

    return score
