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
from robocerebra_rl.rewards import GeminiRewardCache, symbolic_dense_reward
from robocerebra_rl.trace import ToolTraceLogger
from robocerebra_rl.world import ACTIONS, BreakfastTrayWorld, SceneConfig


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
        scene = _scene_from_spec(spec.get("scene"))
        self.world = BreakfastTrayWorld(
            seed=int(spec.get("seed", 0)),
            horizon_ticks=int(spec.get("horizon_ticks", 1000)),
            max_macro_steps=int(spec.get("max_macro_steps", 30 if scene != SceneConfig() else 18)),
            scene=scene,
        )
        self.current_subgoal = self.world.expected_action
        self.reward_cache = GeminiRewardCache(".openreward/gemini_reward_cache.json")
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
        seeds = {"train": [1, 2, 3], "validation": [1001], "test": [2001, 2002, 2003, 2004]}.get(split, [0])
        return [
            {
                "task_id": f"breakfast-tray-{seed}",
                "seed": seed,
                "horizon_ticks": 1000 + (seed % 2) * 500,
                "max_macro_steps": 18 if seed >= 1000 else 30,
                "scene": (
                    SceneConfig.from_seed(seed).as_dict()
                    if seed < 1000
                    else SceneConfig(
                        **{
                            **SceneConfig.from_seed(seed).as_dict(),
                            "distractor_count": max(2, SceneConfig.from_seed(seed).distractor_count),
                            "action_failure_prob": max(0.18, SceneConfig.from_seed(seed).action_failure_prob),
                            "disturbance_severity": max(0.7, SceneConfig.from_seed(seed).disturbance_severity),
                        }
                    ).as_dict()
                ),
                "instruction": (
                    "Prepare and deliver a breakfast tray under a mid-task disturbance. "
                    "Use macro-skills and ask for progress scores at subgoal boundaries."
                ),
            }
            for seed in seeds
        ]

    def get_prompt(self) -> list[TextBlock]:
        return [
            TextBlock(
                text=(
                    "You are controlling a long-horizon RoboCerebra-style manipulation workflow. "
                    "Complete the breakfast tray task by choosing semantic subgoals, executing "
                    "macro-skills, and using dense reward feedback to recover from disturbances."
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
        score = self.reward_cache.score(
            self.world.task.task_id,
            self.world.state_hash(),
            params.subgoal,
            self.world.action_history[-1] if self.world.action_history else "observe",
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
            metadata=dict(score),
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


def _scene_from_spec(value: object) -> SceneConfig:
    if not isinstance(value, dict):
        return SceneConfig()
    return SceneConfig(
        mug_position=tuple(value.get("mug_position", SceneConfig().mug_position)),  # type: ignore[arg-type]
        snack_position=tuple(value.get("snack_position", SceneConfig().snack_position)),  # type: ignore[arg-type]
        tray_position=tuple(value.get("tray_position", SceneConfig().tray_position)),  # type: ignore[arg-type]
        disturbance_tick=int(value.get("disturbance_tick", SceneConfig().disturbance_tick)),
        distractor_count=int(value.get("distractor_count", SceneConfig().distractor_count)),
        action_failure_prob=float(value.get("action_failure_prob", SceneConfig().action_failure_prob)),
        disturbance_severity=float(value.get("disturbance_severity", SceneConfig().disturbance_severity)),
    )


def create_server() -> Server:
    return Server([RoboCerebraRewardLabEnv])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_server().app, host="0.0.0.0", port=8080)
