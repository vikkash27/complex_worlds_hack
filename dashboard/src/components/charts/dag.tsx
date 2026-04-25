import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import { COLORS } from "./theme";

type Kind = "nominal" | "recover" | "disturb" | "goal" | "wrong";

const NODES: { name: string; x: number; y: number; kind: Kind; stage?: string }[] = [
  { name: "Locate items", x: 60, y: 220, kind: "nominal", stage: "Locate" },
  { name: "Clear workspace", x: 200, y: 170, kind: "nominal", stage: "Clear" },
  { name: "Pick mug", x: 340, y: 110, kind: "nominal", stage: "PickMug" },
  { name: "Fill drink", x: 480, y: 95, kind: "nominal", stage: "Fill" },
  { name: "Place snack", x: 600, y: 170, kind: "nominal", stage: "Snack" },
  { name: "Tray nudged", x: 690, y: 300, kind: "disturb" },
  { name: "Recover (replan)", x: 770, y: 240, kind: "recover", stage: "Recover" },
  { name: "Deliver tray", x: 920, y: 170, kind: "goal", stage: "Deliver" },
  { name: "Pick snack first", x: 340, y: 310, kind: "wrong" },
  { name: "Drop mug", x: 480, y: 360, kind: "wrong" },
];

const EDGES: { s: string; t: string; r: number; curve?: number; dashed?: boolean }[] = [
  { s: "Locate items", t: "Clear workspace", r: 0.4 },
  { s: "Clear workspace", t: "Pick mug", r: 0.45 },
  { s: "Pick mug", t: "Fill drink", r: 0.5 },
  { s: "Fill drink", t: "Place snack", r: 0.4 },
  { s: "Place snack", t: "Tray nudged", r: 0.0 },
  { s: "Tray nudged", t: "Recover (replan)", r: 0.6 },
  { s: "Recover (replan)", t: "Deliver tray", r: 0.9 },
  { s: "Locate items", t: "Pick snack first", r: -0.25 },
  { s: "Pick snack first", t: "Drop mug", r: -0.4 },
  { s: "Pick mug", t: "Deliver tray", r: 0.7, curve: 0.35, dashed: true },
];

const KIND_COLOR: Record<Kind, string> = {
  nominal: COLORS.cyan,
  recover: COLORS.warn,
  disturb: COLORS.violet,
  goal: COLORS.good,
  wrong: COLORS.bad,
};

export function SubgoalDAG({ activeStage }: { activeStage?: string }) {
  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: {
      backgroundColor: "#0d0f14",
      borderColor: COLORS.line,
      textStyle: { color: COLORS.text, fontFamily: "JetBrains Mono", fontSize: 11 },
    },
    series: [
      {
        type: "graph",
        layout: "none",
        roam: true,
        draggable: true,
        zoom: 0.95,
        center: [490, 220],
        symbol: "roundRect",
        symbolSize: [120, 36],
        label: {
          show: true,
          color: COLORS.text,
          fontFamily: "Inter",
          fontWeight: 500,
          fontSize: 11,
        },
        edgeSymbol: ["none", "arrow"],
        edgeSymbolSize: [0, 7],
        data: NODES.map((n) => {
          const isActive = n.stage && n.stage === activeStage;
          const base = KIND_COLOR[n.kind];
          return {
            name: n.name,
            x: n.x,
            y: n.y,
            symbolSize: isActive ? [128, 40] : [120, 36],
            itemStyle: {
              color: isActive ? base + "33" : base + "14",
              borderColor: base,
              borderWidth: isActive ? 2 : 1.25,
              shadowBlur: isActive ? 18 : 0,
              shadowColor: isActive ? base : "transparent",
            },
            label: {
              fontWeight: isActive ? 600 : 500,
              color: isActive ? "#ffffff" : COLORS.text,
            },
          };
        }),
        links: EDGES.map((e) => ({
          source: e.s,
          target: e.t,
          lineStyle: {
            color: e.r >= 0 ? COLORS.good : COLORS.bad,
            width: 1 + Math.abs(e.r) * 3,
            opacity: 0.55,
            curveness: e.curve ?? 0.05,
            type: e.dashed ? "dashed" : "solid",
          },
          label: {
            show: true,
            formatter: (e.r >= 0 ? "+" : "") + e.r.toFixed(2),
            color: e.r >= 0 ? COLORS.good : COLORS.bad,
            fontFamily: "JetBrains Mono",
            fontSize: 9,
            opacity: 0.85,
          },
        })),
      },
    ],
  }), [activeStage]);

  return (
    <ReactECharts
      option={option}
      notMerge={false}
      lazyUpdate
      style={{ height: "100%", width: "100%" }}
    />
  );
}
