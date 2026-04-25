from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Iterable

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
from robocerebra_rl.rewards import GeminiRewardCache, resolve_vlm_scorer
from robocerebra_rl.shift import (
    SHIFT_TOOL_NAMES,
    ShiftSpec,
    ShiftWorld,
    build_shift_spec,
    shift_spec_from_dict,
)
from robocerebra_rl.trace import ToolTraceLogger
from robocerebra_rl.world import (
    ACTIONS,
    list_openreward_shift_tasks_for_split,
)


# ----- Pydantic input models --------------------------------------------------


class ChooseSubgoalInput(BaseModel):
    subgoal: str = Field(description="The semantic subgoal the agent intends to complete next.")


class ExecuteSkillInput(BaseModel):
    action: str = Field(description=f"One macro-action from: {', '.join(ACTIONS)}")


class ScoreProgressInput(BaseModel):
    subgoal: str = Field(description="The subgoal to judge against the current state.")


class PlanCreateInput(BaseModel):
    steps: list[str] = Field(description="Ordered list of plan step labels for the active job.")


class PlanReviseInput(BaseModel):
    steps: list[str] = Field(description="Replacement plan steps after a non-stationary event.")


class MemoryWriteInput(BaseModel):
    key: str
    value: str


class MemoryReadInput(BaseModel):
    key: str


class MemorySearchInput(BaseModel):
    query: str


class InventoryCheckInput(BaseModel):
    item: str | None = None


class InventoryConsumeInput(BaseModel):
    item: str
    qty: int = 1


class InventoryRestockInput(BaseModel):
    item: str
    qty: int = 2


class AcknowledgeEventInput(BaseModel):
    event_id: str


class LogJobInput(BaseModel):
    summary: str


# ----- The env --------------------------------------------------------------


class RoboCerebraShiftEnv(Environment):
    """Long-horizon multi-job *shift* environment for OpenReward.

    Each task is a full hospitality shift. An agent must read tickets, plan,
    track inventory, recall guest preferences via memory tools, handle
    deterministic non-stationary events, and finally submit a memory summary
    to win. Episodes routinely require **hundreds to thousands of tool calls**.
    """

    def __init__(self, task_spec: dict[str, Any] | None = None, secrets: dict[str, str] | None = None):
        load_dotenv(override=False)
        super().__init__(task_spec or {}, secrets or {})
        spec = task_spec or {}
        shift_spec = self._resolve_shift_spec(spec)
        self.shift = ShiftWorld(spec=shift_spec)
        cache_path = os.getenv("ROBOCEREBRA_REWARD_CACHE", ".openreward/gemini_reward_cache.json")
        self.reward_cache = GeminiRewardCache(cache_path, scorer=resolve_vlm_scorer())
        trace_path = spec.get("trace_path") or os.getenv("ROBOCEREBRA_TRACE_PATH")
        if not trace_path:
            trace_path = f".openreward/traces/{shift_spec.shift_id}.jsonl"
        self.trace = ToolTraceLogger(Path(str(trace_path)), run_id=shift_spec.shift_id)

    @staticmethod
    def _resolve_shift_spec(spec: dict[str, Any]) -> ShiftSpec:
        if "jobs" in spec and "shift_id" in spec:
            return shift_spec_from_dict(spec)
        split = str(spec.get("split", "train"))
        seed = int(spec.get("seed", 0))
        return build_shift_spec(split=split, seed=seed)

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
        return list_openreward_shift_tasks_for_split(split)

    def get_prompt(self) -> list[TextBlock]:
        spec = self.shift.spec
        jobs_preview = ", ".join(j.task_name for j in spec.jobs[:6])
        return [
            TextBlock(
                text=(
                    "You are running a long-horizon multi-job hospitality shift. "
                    f"Shift `{spec.shift_id}` ({spec.split} split, {len(spec.jobs)} jobs, "
                    f"{len(spec.events)} non-stationary events scheduled). "
                    "Use `read_ticket` to get the next job, `plan_create` before executing, "
                    "`memory_write/read/search/summarize` for guest recall, "
                    "`inventory_check/consume/restock` for resources, and "
                    "`acknowledge_event` + `plan_revise` to handle disturbances. "
                    "End with `submit_done` only after every job is logged, every event "
                    "is acknowledged, and a `memory_summarize` succeeds. "
                    f"First few jobs: {jobs_preview}. "
                    f"Tool budget: {spec.max_tool_calls}. Tool surface: {', '.join(SHIFT_TOOL_NAMES)}."
                )
            )
        ]

    # ----- per-job execution tools -----------------------------------------

    @tool
    async def observe(self) -> ToolOutput:
        overview = self.shift.observe()
        self._record_trace("observe", None, overview, 0.0, {}, "Shift observation refreshed.", self.shift.done)
        blocks: list[TextBlock | ImageBlock] = [TextBlock(text=self._format_overview(overview))]
        if self.shift.current_world is not None:
            blocks.append(self._image_block())
        return ToolOutput(blocks=blocks, reward=0.0, finished=self.shift.done, metadata=overview)

    @tool
    async def choose_subgoal(self, params: ChooseSubgoalInput) -> ToolOutput:
        result = self.shift.choose_subgoal(params.subgoal)
        self._record_trace("choose_subgoal", params.subgoal, self.shift.overview(), 0.0, {}, f"Subgoal `{params.subgoal}` declared.", self.shift.done)
        return ToolOutput(
            blocks=[TextBlock(text=f"Subgoal `{params.subgoal}` accepted={result['accepted']}.")],
            reward=0.0,
            finished=self.shift.done,
            metadata=result,
        )

    @tool
    async def execute_skill(self, params: ExecuteSkillInput) -> ToolOutput:
        result = self.shift.execute_skill(params.action)
        rationale = (
            f"`{params.action}` advanced job by {result.get('progress_delta', 0.0):.3f}."
            if result.get("accepted")
            else f"Execute rejected: {result.get('reason')}"
        )
        self._record_trace(
            "execute_skill",
            params.action,
            self.shift.overview(),
            float(result.get("reward", 0.0)),
            {"progress_delta": float(result.get("progress_delta", 0.0))},
            rationale,
            self.shift.done,
        )
        blocks: list[TextBlock | ImageBlock] = [TextBlock(text=rationale)]
        if self.shift.current_world is not None:
            blocks.append(self._image_block())
        return ToolOutput(
            blocks=blocks,
            reward=float(result.get("reward", 0.0)),
            finished=self.shift.done,
            metadata=result,
        )

    @tool
    async def score_progress(self, params: ScoreProgressInput) -> ToolOutput:
        score = self.shift.score_progress(params.subgoal)
        if self.shift.current_world is not None:
            image_path = self._write_observation_image()
            cached = self.reward_cache.score(
                self.shift.spec.shift_id,
                self.shift.state_hash(),
                params.subgoal,
                self.shift.current_world.action_history[-1] if self.shift.current_world.action_history else "observe",
                progress_delta=float(score["progress_delta"]),
                image_path=image_path,
            )
            score = {**score, **cached, "image_path": image_path}
        reward = float(score.get("progress_delta", 0.0)) + 0.1 * float(score.get("confidence", 0.0))
        self._record_trace(
            "score_progress",
            params.subgoal,
            self.shift.overview(),
            round(reward, 6),
            {"vlm_confidence": float(score.get("confidence", 0.0))},
            str(score.get("rationale", "")),
            self.shift.done,
        )
        return ToolOutput(
            blocks=[TextBlock(text=f"Score for `{params.subgoal}`: {score}")],
            reward=round(reward, 6),
            finished=self.shift.done,
            metadata=score,
        )

    # ----- shift-only tools ------------------------------------------------

    @tool
    async def read_ticket(self) -> ToolOutput:
        result = self.shift.read_ticket()
        ticket = result.get("ticket")
        text = f"Ticket: {ticket}" if ticket else "No more tickets."
        self._record_trace("read_ticket", None, self.shift.overview(), 0.0, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.0, finished=self.shift.done, metadata=result)

    @tool
    async def plan_create(self, params: PlanCreateInput) -> ToolOutput:
        result = self.shift.plan_create(params.steps)
        text = f"plan_create accepted={result.get('accepted')} covers_inventory={result.get('covers_inventory', False)}"
        self._record_trace("plan_create", None, self.shift.overview(), 0.0, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.0, finished=self.shift.done, metadata=result)

    @tool
    async def plan_revise(self, params: PlanReviseInput) -> ToolOutput:
        result = self.shift.plan_revise(params.steps)
        text = f"plan_revise accepted={result.get('accepted')} revisions={result.get('plan_revisions', 0)}"
        self._record_trace("plan_revise", None, self.shift.overview(), 0.05 if result.get("accepted") else 0.0, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.05 if result.get("accepted") else 0.0, finished=self.shift.done, metadata=result)

    @tool
    async def memory_write(self, params: MemoryWriteInput) -> ToolOutput:
        result = self.shift.memory_write(params.key, params.value)
        self._record_trace("memory_write", params.key, self.shift.overview(), 0.0, {}, f"memory[{params.key}] set.", self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=f"Stored {params.key}.")], reward=0.0, finished=self.shift.done, metadata=result)

    @tool
    async def memory_read(self, params: MemoryReadInput) -> ToolOutput:
        result = self.shift.memory_read(params.key)
        text = f"memory[{params.key}] -> {result.get('value')}"
        reward = 0.05 if result.get("found") else 0.0
        self._record_trace("memory_read", params.key, self.shift.overview(), reward, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=reward, finished=self.shift.done, metadata=result)

    @tool
    async def memory_search(self, params: MemorySearchInput) -> ToolOutput:
        result = self.shift.memory_search(params.query)
        hits = result.get("hits", [])
        reward = 0.1 if hits else 0.0
        self._record_trace("memory_search", params.query, self.shift.overview(), reward, {}, f"{len(hits)} hits.", self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=f"memory_search('{params.query}') -> {hits}")], reward=reward, finished=self.shift.done, metadata=result)

    @tool
    async def memory_summarize(self) -> ToolOutput:
        result = self.shift.memory_summarize()
        adequate = bool(result.get("adequate"))
        reward = 0.5 if adequate else 0.0
        self._record_trace("memory_summarize", None, self.shift.overview(), reward, {}, f"adequate={adequate}", self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=f"memory_summarize adequate={adequate}: {result.get('summary')}")], reward=reward, finished=self.shift.done, metadata=result)

    @tool
    async def inventory_check(self, params: InventoryCheckInput) -> ToolOutput:
        result = self.shift.inventory_check(params.item)
        text = f"inventory_check({params.item}) -> {result}"
        self._record_trace("inventory_check", params.item, self.shift.overview(), 0.0, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.0, finished=self.shift.done, metadata=result)

    @tool
    async def inventory_consume(self, params: InventoryConsumeInput) -> ToolOutput:
        result = self.shift.inventory_consume(params.item, params.qty)
        text = f"inventory_consume({params.item}, {params.qty}) -> {result}"
        reward = 0.05 if result.get("consumed") else -0.05
        self._record_trace("inventory_consume", params.item, self.shift.overview(), reward, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=reward, finished=self.shift.done, metadata=result)

    @tool
    async def inventory_restock(self, params: InventoryRestockInput) -> ToolOutput:
        result = self.shift.inventory_restock(params.item, params.qty)
        text = f"inventory_restock({params.item}, {params.qty}) -> {result}"
        self._record_trace("inventory_restock", params.item, self.shift.overview(), 0.05, {}, text, self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.05, finished=self.shift.done, metadata=result)

    @tool
    async def clock_get(self) -> ToolOutput:
        result = self.shift.clock_get()
        self._record_trace("clock_get", None, self.shift.overview(), 0.0, {}, "clock_get", self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=f"clock={result}")], reward=0.0, finished=self.shift.done, metadata=result)

    @tool
    async def acknowledge_event(self, params: AcknowledgeEventInput) -> ToolOutput:
        result = self.shift.acknowledge_event(params.event_id)
        reward = 0.25 if result.get("accepted") else -0.05
        self._record_trace("acknowledge_event", params.event_id, self.shift.overview(), reward, {}, f"event {params.event_id}: {result}", self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=f"acknowledge_event({params.event_id}) -> {result}")], reward=reward, finished=self.shift.done, metadata=result)

    @tool
    async def log_job(self, params: LogJobInput) -> ToolOutput:
        result = self.shift.log_job(params.summary)
        reward = 0.5 if result.get("outcome") in {"completed", "completed_with_issues"} else -0.2
        self._record_trace("log_job", None, self.shift.overview(), reward, {"outcome": 1.0 if reward > 0 else 0.0}, f"log_job -> {result}", self.shift.done)
        return ToolOutput(blocks=[TextBlock(text=f"log_job -> {result}")], reward=reward, finished=self.shift.done, metadata=result)

    @tool
    async def submit_done(self) -> ToolOutput:
        result = self.shift.submit_done()
        reward = 1.0 if result.get("success") else -0.5
        self._record_trace("submit_done", None, self.shift.overview(), reward, {"success": 1.0 if reward > 0 else 0.0}, f"submit_done -> {result}", True)
        text = "Shift complete." if result.get("success") else f"Shift submitted but incomplete: {result}"
        return ToolOutput(blocks=[TextBlock(text=text)], reward=reward, finished=True, metadata=result)

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _format_overview(overview: dict[str, Any]) -> str:
        return (
            f"Shift {overview['shift_id']} | tool_calls={overview['tool_calls']}/{overview['max_tool_calls']} | "
            f"completed={len(overview['completed_jobs'])} remaining={overview['remaining_jobs']} | "
            f"current={overview['current_job']} active_events={overview['active_events']} | "
            f"memory_keys={len(overview['memory_keys'])}"
        )

    def _image_block(self) -> ImageBlock:
        if self.shift.current_world is None:
            buffer = BytesIO()
            from PIL import Image
            Image.new("RGB", (16, 16), (32, 32, 32)).save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            return ImageBlock(data=data, mimeType="image/png")
        image = render_world(self.shift.current_world)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ImageBlock(data=data, mimeType="image/png")

    def _write_observation_image(self) -> str:
        image_dir = Path(os.getenv("ROBOCEREBRA_OBSERVATION_IMAGE_DIR", ".openreward/frames"))
        image_dir.mkdir(parents=True, exist_ok=True)
        if self.shift.current_world is None:
            return ""
        suffix = f"{self.shift.current_world.macro_steps:03d}-{self.shift.state_hash()}"
        image_path = image_dir / f"{self.shift.spec.shift_id}-{suffix}.png"
        render_world(self.shift.current_world, image_path)
        return str(image_path)

    def _record_trace(
        self,
        tool_name: str,
        action: str | None,
        observation: dict[str, Any],
        reward: float,
        reward_components: dict[str, float],
        rationale: str,
        finished: bool,
    ) -> None:
        self.trace.record(
            tool_name=tool_name,
            task_id=self.shift.spec.shift_id,
            action=action,
            observation=observation,
            reward=reward,
            reward_components=reward_components,
            rationale=rationale,
            finished=finished,
            state_hash=self.shift.state_hash(),
        )


# Backwards-compatible alias retained so external imports keep working.
RoboCerebraRewardLabEnv = RoboCerebraShiftEnv


def create_server() -> Server:
    return Server([RoboCerebraShiftEnv])


def iter_tool_names() -> Iterable[str]:
    return SHIFT_TOOL_NAMES


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_server().app, host="0.0.0.0", port=8080)
