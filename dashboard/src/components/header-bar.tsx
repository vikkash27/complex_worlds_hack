import { motion } from "framer-motion";
import type { LiveState } from "@/hooks/use-live-state";
import { STAGES } from "@/hooks/use-live-state";

export function HeaderBar({ live }: { live: LiveState }) {
  return (
    <header className="h-14 border-b border-ink-600/70 flex items-center px-6 gap-8 bg-ink-900/60 backdrop-blur">
      {/* Wordmark */}
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="relative h-2 w-2">
          <span className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-50" />
          <span className="absolute inset-0 rounded-full bg-emerald-400" />
        </div>
        <div className="text-sm font-semibold tracking-tight text-neutral-100">
          RoboCerebra<span className="text-neutral-500"> · </span>Reward Lab
        </div>
        <div className="hidden md:block text-[10px] font-mono tracking-[0.18em] uppercase text-neutral-500 ml-2">
          ep {live.episode.toString().padStart(2, "0")} / 60
        </div>
      </div>

      {/* Stage progress dots */}
      <div className="flex items-center gap-1.5 shrink-0">
        {STAGES.map((s, i) => (
          <div key={s} className="flex items-center gap-1.5">
            <span
              className={
                "h-1.5 w-1.5 rounded-full transition-colors " +
                (i < live.stageIndex
                  ? "bg-emerald-400"
                  : i === live.stageIndex
                  ? "bg-emerald-300 ring-2 ring-emerald-400/30"
                  : "bg-ink-500")
              }
            />
            {i < STAGES.length - 1 && (
              <span
                className={
                  "h-px w-3 " +
                  (i < live.stageIndex ? "bg-emerald-400/60" : "bg-ink-600")
                }
              />
            )}
          </div>
        ))}
        <span className="ml-3 text-[10px] font-mono uppercase tracking-[0.2em] text-neutral-400">
          {live.stage}
        </span>
      </div>

      {/* Live metrics — last tool · tick · reward Σ */}
      <div className="ml-auto flex items-center gap-7">
        <Pill
          label="last tool"
          value={
            <motion.span
              key={live.lastTool + live.toolCalls}
              initial={{ opacity: 0.4 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.16 }}
              className="text-sky-300 text-[13px] font-mono"
            >
              {live.lastTool}
            </motion.span>
          }
        />
        <Pill
          label="tick"
          value={
            <span className="tabular-nums text-neutral-100">
              {live.tick.toString().padStart(4, "0")}
              <span className="text-neutral-600 text-xs">/1001</span>
            </span>
          }
        />
        <Pill
          label="reward Σ"
          value={
            <span
              className={
                "tabular-nums " +
                (live.rewardSum >= 0 ? "text-emerald-300" : "text-rose-400")
              }
            >
              {live.rewardSum >= 0 ? "+" : ""}
              {live.rewardSum.toFixed(2)}
            </span>
          }
        />
      </div>
    </header>
  );
}

function Pill({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col items-end leading-none">
      <div className="font-mono text-[15px] font-medium">{value}</div>
      <div className="text-[9px] uppercase tracking-[0.2em] text-neutral-500 mt-1">
        {label}
      </div>
    </div>
  );
}
