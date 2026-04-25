import { useEffect, useState } from "react";

export type LiveState = {
  tick: number;
  toolCalls: number;
  lastTool: string;
  rewardSum: number;
  stage: string;
  stageIndex: number;
  episode: number;
};

const STAGES = ["Locate", "Clear", "PickMug", "Fill", "Snack", "Recover", "Deliver"];
const TOOLS = ["observe", "choose_subgoal", "execute_skill", "score_progress"];

// Stub. Replace with EventSource('/stream') on the SSE backend.
export function useLiveState(): LiveState {
  const [s, setS] = useState<LiveState>({
    tick: 0, toolCalls: 0, lastTool: "observe",
    rewardSum: 0, stage: STAGES[0], stageIndex: 0, episode: 42,
  });

  useEffect(() => {
    const id = setInterval(() => {
      setS((p) => {
        const tick = Math.min(1001, p.tick + 6 + Math.floor(Math.random() * 5));
        const stageIndex = Math.min(
          STAGES.length - 1,
          Math.floor((tick / 1001) * STAGES.length)
        );
        return {
          tick,
          toolCalls: p.toolCalls + 1,
          lastTool: TOOLS[Math.floor(Math.random() * TOOLS.length)],
          rewardSum: +(p.rewardSum + (Math.random() * 0.4 - 0.04)).toFixed(2),
          stage: STAGES[stageIndex],
          stageIndex,
          episode: p.episode,
        };
      });
    }, 480);
    return () => clearInterval(id);
  }, []);

  return s;
}

export { STAGES };
