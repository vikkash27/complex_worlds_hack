from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from openreward.environments import (
    Environment,
    ImageBlock,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)
from pydantic import BaseModel, Field

from robocerebra_rl.render import render_world
from robocerebra_rl.rewards import GeminiRewardCache, gemini_reward_scorer, symbolic_dense_reward
from robocerebra_rl.trace import ToolTraceLogger
from robocerebra_rl.world import (
    ACTIONS,
    TASK_LIBRARY,
    BreakfastTrayWorld,
    horizon_ticks_for_seed,
    list_openreward_tasks_for_split,
    max_macro_steps_for,
    scene_from_spec_dict,
)


class ChooseSubgoalInput(BaseModel):
    subgoal: str = Field(description="The semantic subgoal the agent intends to complete next.")


class ExecuteSkillInput(BaseModel):
    action: str = Field(description=f"One macro-action from: {', '.join(ACTIONS)}")


class ScoreProgressInput(BaseModel):
    subgoal: str = Field(description="The subgoal to judge against the current state.")


class RoboCerebraRewardLabEnv(Environment):
    def __init__(self, task_spec: dict[str, Any] | None = None, secrets: dict[str, str] | None = None):
        load_dotenv(override=False)
        super().__init__(task_spec or {}, secrets or {})
        spec = task_spec or {}
        seed = int(spec.get("seed", 0))
        task_name = str(spec.get("task_name", "breakfast_tray"))
        scene = scene_from_spec_dict(spec.get("scene"), seed=seed)
        horizon = int(spec.get("horizon_ticks", horizon_ticks_for_seed(seed)))
        max_macro = int(spec.get("max_macro_steps", max_macro_steps_for(task_name, seed)))
        self.world = BreakfastTrayWorld(
            seed=seed,
            horizon_ticks=horizon,
            max_macro_steps=max_macro,
            scene=scene,
            task_name=task_name,
        )
        self.current_subgoal = self.world.expected_action
        scorer = gemini_reward_scorer() if os.getenv("ROBOCEREBRA_USE_GEMINI_VISION") == "1" else None
        cache_path = os.getenv("ROBOCEREBRA_REWARD_CACHE", ".openreward/gemini_reward_cache.json")
        self.reward_cache = GeminiRewardCache(cache_path, scorer=scorer)
        self.last_progress_delta = 0.0
        trace_path = spec.get("trace_path") or os.getenv("ROBOCEREBRA_TRACE_PATH")
        if not trace_path:
            trace_path = f".openreward/traces/{self.world.task.task_id}.jsonl"
        self.trace = ToolTraceLogger(Path(str(trace_path)), run_id=self.world.task.task_id)

    @classmethod
    def name(cls) -> str:
        return "robocerebra_reward_lab"

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="train", type="train"),
            Split(name="validation", type="validation"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict[str, Any]]:
        return list_openreward_tasks_for_split(split)

    def get_prompt(self) -> list[TextBlock]:
        return [
            TextBlock(
                text=(
                    "You are controlling a long-horizon RoboCerebra-style manipulation workflow. "
                    f"Task: {self.world.task.label}. Instruction: {self.world.task.instruction} "
                    "Choose semantic subgoals, execute macro-skills, and use dense reward feedback "
                    "to recover from failures when the task requires it."
                )
            )
        ]

    @tool
    async def observe(self) -> ToolOutput:
        observation = self.world.observe()
        self._record_trace(
            "observe",
            None,
            observation,
            0.0,
            {},
            "Observation returned current progress, expected next subgoal, and rendered frame.",
            self.world.done,
        )
        return ToolOutput(
            blocks=[
                TextBlock(text=self._format_observation()),
                self._image_block(),
            ],
            reward=0.0,
            finished=self.world.done,
            metadata=observation,
        )

    @tool
    async def choose_subgoal(self, params: ChooseSubgoalInput) -> ToolOutput:
        self.current_subgoal = params.subgoal
        self._record_trace(
            "choose_subgoal",
            params.subgoal,
            self.world.observe(),
            0.0,
            {},
            f"Agent declared intended subgoal `{params.subgoal}` before acting.",
            self.world.done,
        )
        return ToolOutput(
            blocks=[TextBlock(text=f"Subgoal set to `{self.current_subgoal}`.")],
            reward=0.0,
            finished=self.world.done,
            metadata={"current_subgoal": self.current_subgoal},
        )

    @tool
    async def execute_skill(self, params: ExecuteSkillInput) -> ToolOutput:
        transition = self.world.step(params.action)
        self.last_progress_delta = transition.progress_delta
        reward = symbolic_dense_reward(transition)
        reward_components = {
            "progress": round(transition.progress_delta * 1.5, 6),
            "stage_bonus": round(transition.progress_after * 0.15 if transition.progress_delta > 0 else 0.0, 6),
            "success": 1.0 if transition.success else 0.0,
        }
        rationale = (
            f"`{params.action}` advanced the workflow by {transition.progress_delta:.3f}."
            if transition.progress_delta > 0
            else f"`{params.action}` did not advance: {transition.observation.get('last_failure_reason') or 'not expected now'}."
        )
        self._record_trace(
            "execute_skill",
            params.action,
            transition.observation,
            reward,
            reward_components,
            rationale,
            transition.done,
            transition.state_hash,
        )
        return ToolOutput(
            blocks=[
                TextBlock(
                    text=(
                        f"Executed `{params.action}`. Expected `{transition.expected_action}`. "
                        f"Progress is now {transition.progress_after:.0%} after {transition.ticks} ticks."
                    )
                ),
                self._image_block(),
            ],
            reward=reward,
            finished=transition.done,
            metadata={
                **transition.observation,
                "action": params.action,
                "progress_delta": transition.progress_delta,
                "symbolic_dense_reward": reward,
                "completed_subgoals": transition.observation["completed_subgoals"],
            },
        )

    @tool
    async def score_progress(self, params: ScoreProgressInput) -> ToolOutput:
        image_path = self._write_observation_image()
        score = self.reward_cache.score(
            self.world.task.task_id,
            self.world.state_hash(),
            params.subgoal,
            self.world.action_history[-1] if self.world.action_history else "observe",
            progress_delta=self.last_progress_delta,
            image_path=image_path,
        )
        reward = float(score["progress_delta"]) + 0.1 * float(score["confidence"])
        self._record_trace(
            "score_progress",
            params.subgoal,
            self.world.observe(),
            round(reward, 6),
            {
                "vlm_progress_delta": float(score["progress_delta"]),
                "vlm_confidence_bonus": 0.1 * float(score["confidence"]),
            },
            str(score["rationale"]),
            self.world.done,
        )
        return ToolOutput(
            blocks=[
                TextBlock(
                    text=(
                        f"Gemini reward score for `{params.subgoal}`: "
                        f"delta={score['progress_delta']}, confidence={score['confidence']}. "
                        f"{score['rationale']}"
                    )
                )
            ],
            reward=round(reward, 6),
            finished=self.world.done,
            metadata={**dict(score), "image_path": image_path},
        )

    @tool
    async def submit_done(self) -> ToolOutput:
        finished = self.world.success
        reward = 1.0 if finished else -0.5
        self.world.done = True
        self._record_trace(
            "submit_done",
            "submit_done",
            self.world.observe(),
            reward,
            {"success": reward},
            "Final submission checks whether all subgoals and disturbance recovery are complete.",
            True,
        )
        return ToolOutput(
            blocks=[TextBlock(text="Task complete." if finished else "Task submitted before completion.")],
            reward=reward,
            finished=True,
            metadata=self.world.observe(),
        )

    def _format_observation(self) -> str:
        observation = self.world.observe()
        return (
            f"Instruction: {observation['instruction']}\n"
            f"Ticks: {observation['ticks']} / {observation['horizon_ticks']}\n"
            f"Completed: {observation['completed_subgoals']}\n"
            f"Remaining: {observation['remaining_subgoals']}\n"
            f"Expected next: {observation['expected_next']}"
        )

    def _image_block(self) -> ImageBlock:
        image = render_world(self.world)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ImageBlock(data=data, mimeType="image/png")

    def _write_observation_image(self) -> str:
        image_dir = Path(os.getenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", ".openreward/frames"))
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{self.world.task.task_id}-{self.world.macro_steps:03d}-{self.world.state_hash()}.png"
        render_world(self.world, image_path)
        return str(image_path)

    def _record_trace(
        self,
        tool_name: str,
        action: str | None,
        observation: dict[str, object],
        reward: float,
        reward_components: dict[str, float],
        rationale: str,
        finished: bool,
        state_hash: str | None = None,
    ) -> None:
        self.trace.record(
            tool_name=tool_name,
            task_id=self.world.task.task_id,
            action=action,
            observation=observation,
            reward=reward,
            reward_components=reward_components,
            rationale=rationale,
            finished=finished,
            state_hash=state_hash or self.world.state_hash(),
        )


def create_server() -> Server:
    return Server([RoboCerebraRewardLabEnv])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_server().app, host="0.0.0.0", port=8080)
