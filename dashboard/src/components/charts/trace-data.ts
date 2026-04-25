// Synthetic trace data for the breakfast-tray episode.
// Replace with live SSE feed in production (see /stream wiring).

export const LANES = [
  "observe",
  "choose_subgoal",
  "execute_skill",
  "score_progress",
  "submit_done",
] as const;

export const STAGES = [
  "Locate",
  "Clear",
  "PickMug",
  "Fill",
  "Snack",
  "Recover",
  "Deliver",
];

export type ToolCall = {
  tool: (typeof LANES)[number];
  start: number;
  end: number;
  reward: number;
  stage: string;
};

const STAGE_BOUNDARIES = [0, 140, 260, 400, 540, 620, 820, 1001];
const RNG = mulberry32(42);

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildEpisode(): ToolCall[] {
  const calls: ToolCall[] = [];
  STAGES.forEach((stg, si) => {
    const start = STAGE_BOUNDARIES[si];
    const end = STAGE_BOUNDARIES[si + 1];

    calls.push({ tool: "choose_subgoal", start, end: start + 8, reward: 0.05, stage: stg });

    const observes = 3 + Math.floor(RNG() * 3);
    for (let k = 0; k < observes; k++) {
      const s = start + 10 + k * Math.floor((end - start - 20) / 4);
      calls.push({ tool: "observe", start: s, end: s + 6, reward: 0, stage: stg });
    }

    let cursor = start + 18;
    while (cursor < end - 30) {
      const dur = 30 + Math.floor(RNG() * 45);
      const r = stg === "Recover" ? 0.35 : 0.15 + RNG() * 0.4;
      calls.push({
        tool: "execute_skill",
        start: cursor,
        end: Math.min(cursor + dur, end - 12),
        reward: r,
        stage: stg,
      });
      calls.push({
        tool: "score_progress",
        start: cursor + dur - 4,
        end: cursor + dur + 6,
        reward: 0.05,
        stage: stg,
      });
      cursor += dur + 8;
    }

    if (stg === "Recover") {
      calls.push({ tool: "execute_skill", start: start + 8, end: start + 24, reward: -0.2, stage: stg });
      calls.push({ tool: "choose_subgoal", start: start + 25, end: start + 33, reward: 0.05, stage: stg });
    }
  });
  calls.push({ tool: "submit_done", start: 992, end: 1001, reward: 1.0, stage: "Deliver" });
  return calls;
}

export const HEAT_BINS = 100;
export function buildHeat(calls: ToolCall[]) {
  const heat = new Array(HEAT_BINS).fill(0) as number[];
  calls.forEach((c) => {
    const b = Math.floor(((c.start + c.end) / 2 / 1001) * HEAT_BINS);
    heat[b] = (heat[b] || 0) + c.reward;
  });
  return heat;
}
