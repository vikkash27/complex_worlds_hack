import ReactECharts from "echarts-for-react";
import { COLORS } from "./theme";

const NODES = [
  { name: "Locate" },
  { name: "Clear" },
  { name: "PickMug" },
  { name: "Fill" },
  { name: "Snack" },
  { name: "Recover" },
  { name: "Deliver" },
  { name: "observe" },
  { name: "choose_subgoal" },
  { name: "execute_skill" },
  { name: "score_progress" },
  { name: "submit_done" },
  { name: "success" },
  { name: "partial" },
  { name: "fail" },
];

const STAGE_TO_TOOL: [string, string, number][] = [
  ["Locate", "observe", 60], ["Locate", "choose_subgoal", 20], ["Locate", "execute_skill", 40],
  ["Clear", "observe", 40], ["Clear", "execute_skill", 80], ["Clear", "score_progress", 20],
  ["PickMug", "observe", 30], ["PickMug", "execute_skill", 120], ["PickMug", "score_progress", 30],
  ["Fill", "execute_skill", 110], ["Fill", "score_progress", 30], ["Fill", "observe", 20],
  ["Snack", "execute_skill", 90], ["Snack", "observe", 30], ["Snack", "score_progress", 20],
  ["Recover", "choose_subgoal", 40], ["Recover", "execute_skill", 60], ["Recover", "score_progress", 20],
  ["Deliver", "execute_skill", 140], ["Deliver", "submit_done", 60],
];

const TOOL_TO_OUTCOME: [string, string, number][] = [
  ["observe", "success", 150], ["observe", "partial", 30],
  ["choose_subgoal", "success", 50], ["choose_subgoal", "partial", 10],
  ["execute_skill", "success", 520], ["execute_skill", "partial", 80], ["execute_skill", "fail", 40],
  ["score_progress", "success", 110], ["score_progress", "partial", 10],
  ["submit_done", "success", 60],
];

const LINKS = [...STAGE_TO_TOOL, ...TOOL_TO_OUTCOME].map(([source, target, value]) => ({
  source,
  target,
  value,
}));

export function CreditSankey() {
  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      backgroundColor: "#0d0f14",
      borderColor: COLORS.line,
      textStyle: { color: COLORS.text, fontFamily: "JetBrains Mono" },
    },
    series: [
      {
        type: "sankey",
        data: NODES,
        links: LINKS,
        left: 20,
        right: 120,
        top: 12,
        bottom: 12,
        nodeWidth: 12,
        nodeGap: 10,
        label: {
          color: "#ffffff",
          fontFamily: "Inter",
          fontSize: 11,
          fontWeight: 600,
        },
        itemStyle: { borderWidth: 0, color: "#ffffff" },
        lineStyle: { color: "#ffffff", opacity: 0.55, curveness: 0.55 },
        emphasis: {
          focus: "adjacency",
          lineStyle: { opacity: 0.95, color: "#ffffff" },
          itemStyle: { color: COLORS.good },
        },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
