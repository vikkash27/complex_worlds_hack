import { motion } from "framer-motion";
import { AuroraBg } from "@/components/ui/aurora-bg";
import { Card } from "@/components/ui/card";
import { HeaderBar } from "@/components/header-bar";
import { SimPanel } from "@/components/sim-panel";
import { SubgoalDAG } from "@/components/charts/dag";
import { useLiveState } from "@/hooks/use-live-state";

export default function App() {
  const live = useLiveState();

  return (
    <div className="relative h-screen w-screen overflow-hidden flex flex-col">
      <AuroraBg />

      <HeaderBar live={live} />

      <main
        className="flex-1 min-h-0 grid gap-3 p-3"
        style={{ gridTemplateColumns: "58fr 42fr" }}
      >
        <Card
          title="Isaac Sim"
          subtitle="rollout · 1001-tick horizon"
          trailing={
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-neutral-500">
              16:9 · live
            </span>
          }
        >
          <SimPanel stage={live.stage} />
        </Card>

        <Card
          title="Subgoal Plan"
          subtitle="commitment DAG · downstream credit"
          trailing={
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-emerald-300">
              active · {live.stage}
            </span>
          }
        >
          <div className="flex flex-col h-full">
            {/* Big total-tool-calls KPI above the DAG canvas */}
            <div className="flex items-end justify-between px-5 pt-4 pb-3 border-b border-ink-600/40">
              <div className="flex items-baseline gap-3">
                <motion.div
                  key={live.toolCalls}
                  initial={{ opacity: 0.45, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                  className="font-mono text-[44px] leading-none font-semibold text-emerald-300 tabular-nums tracking-tight"
                >
                  {live.toolCalls.toString().padStart(4, "0")}
                </motion.div>
                <div className="flex flex-col leading-tight">
                  <span className="text-[10px] uppercase tracking-[0.22em] text-neutral-400">
                    total tool calls
                  </span>
                  <span className="text-[10px] font-mono text-neutral-600 mt-0.5">
                    horizon Δ {live.tick}/1001
                  </span>
                </div>
              </div>
              <div className="flex flex-col items-end leading-tight">
                <span className="text-[10px] uppercase tracking-[0.22em] text-neutral-400">
                  episode
                </span>
                <span className="font-mono text-sm text-neutral-200 tabular-nums mt-0.5">
                  {live.episode.toString().padStart(2, "0")}
                  <span className="text-neutral-600"> / 60</span>
                </span>
              </div>
            </div>

            <div className="flex-1 min-h-0">
              <SubgoalDAG activeStage={live.stage} />
            </div>
          </div>
        </Card>
      </main>
    </div>
  );
}
