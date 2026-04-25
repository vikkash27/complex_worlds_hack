export const COLORS = {
  bg: "#08090c",
  panel: "#10131a",
  line: "#1e2330",
  text: "#e7eaf3",
  muted: "#7a8095",
  dim: "#4a5061",
  good: "#7cf2c4",
  cyan: "#5ac8fa",
  warn: "#ffb86b",
  bad: "#ff6b8b",
  violet: "#9b5bff",
};

export function rewardColor(r: number) {
  if (r < 0) return COLORS.bad;
  if (r < 0.2) return COLORS.warn;
  if (r < 0.5) return COLORS.good;
  return COLORS.cyan;
}
