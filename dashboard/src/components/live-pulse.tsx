import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const TOOLS = ["observe", "choose_subgoal", "execute_skill", "score_progress"];

export function LivePulse() {
  const [count, setCount] = useState(0);
  const [tick, setTick] = useState(0);
  const [lastTool, setLastTool] = useState("observe");
  const [reward, setReward] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setCount((c) => c + 1);
      setTick((t) => Math.min(1001, t + 4 + Math.floor(Math.random() * 6)));
      setLastTool(TOOLS[Math.floor(Math.random() * TOOLS.length)]);
      setReward((r) => +(r + (Math.random() * 0.5 - 0.05)).toFixed(2));
    }, 480);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="h-full flex items-center justify-between px-5 gap-6">
      <div className="flex items-center gap-3 text-[10px] font-mono uppercase tracking-[0.22em] text-emerald-300">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
        </span>
        live · sse stream
      </div>

      <div className="flex items-center gap-8 ml-auto">
        <Metric label="tool calls" highlight>
          <motion.span
            key={count}
            initial={{ opacity: 0.4, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.18 }}
            className="font-mono text-2xl text-white tracking-tight tabular-nums"
          >
            {count.toString().padStart(4, "0")}
          </motion.span>
        </Metric>
        <Metric label="tick">
          <span className="font-mono text-lg text-neutral-200 tabular-nums">
            {tick.toString().padStart(4, "0")}
          </span>
          <span className="font-mono text-[10px] text-neutral-600 ml-1">/ 1001</span>
        </Metric>
        <Metric label="last tool">
          <motion.span
            key={lastTool + count}
            initial={{ opacity: 0.4 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.18 }}
            className="font-mono text-sm text-sky-300"
          >
            {lastTool}
          </motion.span>
        </Metric>
        <Metric label="reward Σ">
          <span className={"font-mono text-lg tabular-nums " + (reward >= 0 ? "text-emerald-300" : "text-rose-400")}>
            {reward >= 0 ? "+" : ""}{reward.toFixed(2)}
          </span>
        </Metric>
        <Metric label="episode">
          <span className="font-mono text-lg text-neutral-200 tabular-nums">42<span className="text-[10px] text-neutral-600"> / 60</span></span>
        </Metric>
      </div>
    </div>
  );
}

function Metric({
  label,
  children,
  highlight,
}: {
  label: string;
  children: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <div className="flex flex-col items-start leading-none">
      <div className={"flex items-baseline " + (highlight ? "" : "")}>{children}</div>
      <div className="text-[9px] font-mono uppercase tracking-[0.18em] text-neutral-500 mt-1">
        {label}
      </div>
    </div>
  );
}
