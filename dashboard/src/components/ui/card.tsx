import { cn } from "@/lib/utils";
import { ReactNode } from "react";

export function Card({
  className,
  children,
  title,
  subtitle,
  trailing,
}: {
  className?: string;
  children?: ReactNode;
  title?: string;
  subtitle?: string;
  trailing?: ReactNode;
}) {
  return (
    <div
      className={cn(
        "relative rounded-xl border border-ink-600/60 bg-ink-900/50 overflow-hidden flex flex-col",
        "ring-1 ring-inset ring-white/[0.02]",
        className
      )}
    >
      {(title || trailing) && (
        <div className="flex items-center justify-between px-4 h-10 border-b border-ink-600/50 shrink-0">
          <div className="flex items-baseline gap-3">
            <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-neutral-300">
              {title}
            </span>
            {subtitle && (
              <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-neutral-500">
                {subtitle}
              </span>
            )}
          </div>
          {trailing}
        </div>
      )}
      <div className="flex-1 min-h-0 relative">{children}</div>
    </div>
  );
}
