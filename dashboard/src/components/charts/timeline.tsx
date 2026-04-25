import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import { COLORS, rewardColor } from "./theme";
import { buildEpisode, buildHeat, HEAT_BINS, LANES } from "./trace-data";

export function Timeline() {
  const calls = useMemo(() => buildEpisode(), []);
  const heat = useMemo(() => buildHeat(calls), [calls]);

  const ganttData = calls.map((c) => ({
    name: `${c.stage} · ${c.tool}`,
    value: [LANES.indexOf(c.tool), c.start, c.end, c.reward, c.stage],
    itemStyle: {
      color: rewardColor(c.reward),
      borderRadius: 3,
      borderColor: "#0c0e13",
      borderWidth: 1,
    },
  }));

  const ganttOption = {
    backgroundColor: "transparent",
    grid: { left: 120, right: 20, top: 10, bottom: 40 },
    tooltip: {
      trigger: "item",
      backgroundColor: "#0d0f14",
      borderColor: COLORS.line,
      textStyle: { color: COLORS.text, fontFamily: "JetBrains Mono" },
      formatter: (p: any) =>
        `<b>${p.value[4]} · ${LANES[p.value[0]]}</b><br/>tick ${p.value[1]} → ${p.value[2]}<br/>reward Δ ${p.value[3].toFixed(3)}`,
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1001,
      axisLine: { lineStyle: { color: COLORS.dim } },
      axisLabel: { color: COLORS.muted, fontFamily: "JetBrains Mono", fontSize: 10 },
      splitLine: { lineStyle: { color: COLORS.line, type: "dashed" } },
    },
    yAxis: {
      type: "category",
      data: LANES as unknown as string[],
      inverse: true,
      axisLine: { lineStyle: { color: COLORS.dim } },
      axisTick: { show: false },
      axisLabel: {
        color: COLORS.text,
        fontFamily: "JetBrains Mono",
        fontSize: 11,
        fontWeight: 500,
      },
      splitLine: { show: true, lineStyle: { color: COLORS.line } },
    },
    dataZoom: [
      {
        type: "slider",
        xAxisIndex: 0,
        height: 10,
        bottom: 10,
        backgroundColor: "#0a0c11",
        fillerColor: "rgba(124,242,196,.12)",
        borderColor: COLORS.line,
        handleStyle: { color: COLORS.cyan },
        textStyle: { color: COLORS.muted, fontSize: 10 },
      },
      { type: "inside", xAxisIndex: 0 },
    ],
    series: [
      {
        type: "custom",
        renderItem: (_params: any, api: any) => {
          const cat = api.value(0);
          const s = api.coord([api.value(1), cat]);
          const e = api.coord([api.value(2), cat]);
          const h = api.size([0, 1])[1] * 0.55;
          const w = Math.max(2, e[0] - s[0]);
          return {
            type: "rect",
            shape: { x: s[0], y: s[1] - h / 2, width: w, height: h },
            style: api.style({ stroke: "#0c0e13" }),
          };
        },
        encode: { x: [1, 2], y: 0 },
        data: ganttData,
        markLine: {
          symbol: "none",
          label: {
            formatter: "▲ disturbance",
            color: COLORS.warn,
            fontFamily: "JetBrains Mono",
            fontSize: 10,
            position: "insideEndTop",
          },
          lineStyle: { color: COLORS.warn, type: "dashed", width: 1.5 },
          data: [{ xAxis: 620 }],
        },
      },
    ],
  };

  const heatOption = {
    backgroundColor: "transparent",
    grid: { left: 120, right: 20, top: 4, bottom: 12 },
    tooltip: {
      trigger: "item",
      backgroundColor: "#0d0f14",
      borderColor: COLORS.line,
      textStyle: { color: COLORS.text, fontFamily: "JetBrains Mono" },
      formatter: (p: any) =>
        `bin ${p.value[0]} · reward Σ ${(p.value[2] as number).toFixed(3)}`,
    },
    xAxis: { type: "category", data: [...Array(HEAT_BINS).keys()], show: false },
    yAxis: {
      type: "category",
      data: ["reward Δ"],
      axisLine: { lineStyle: { color: COLORS.dim } },
      axisTick: { show: false },
      axisLabel: { color: COLORS.muted, fontFamily: "JetBrains Mono", fontSize: 10 },
    },
    visualMap: {
      min: -0.3,
      max: 1.0,
      show: false,
      inRange: { color: ["#1a1d27", "#2a2f3d", COLORS.warn, COLORS.good, COLORS.cyan] },
    },
    series: [
      {
        type: "heatmap",
        data: heat.map((v, i) => [i, 0, v]),
        itemStyle: { borderColor: "#0a0c11", borderWidth: 1 },
      },
    ],
  };

  return (
    <div className="w-full h-full flex flex-col">
      <ReactECharts option={ganttOption} style={{ height: "78%", width: "100%" }} />
      <ReactECharts option={heatOption} style={{ height: "22%", width: "100%" }} />
    </div>
  );
}
