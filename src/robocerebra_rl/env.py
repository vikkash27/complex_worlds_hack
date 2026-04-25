from __future__ import annotations

import base64
from io import BytesIO
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
from robocerebra_rl.world import ACTIONS, BreakfastTrayWorld


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
        self.world = BreakfastTrayWorld(
            seed=int(spec.get("seed", 0)),
            horizon_ticks=int(spec.get("horizon_ticks", 1000)),
        )
        self.current_subgoal = self.world.expected_action
        self.reward_cache = GeminiRewardCache(".openreward/gemini_reward_cache.json")

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
        seeds = {"train": [1, 2, 3], "validation": [101], "test": [201, 202]}.get(split, [0])
        return [
            {
                "task_id": f"breakfast-tray-{seed}",
                "seed": seed,
                "horizon_ticks": 1000 + (seed % 2) * 500,
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


def create_server() -> Server:
    return Server([RoboCerebraRewardLabEnv])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_server().app, host="0.0.0.0", port=8080)
