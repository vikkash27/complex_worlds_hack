import { cn } from "@/lib/utils";

// Subtle, full-app aurora layer. Pin behind everything with `fixed inset-0`.
// Sim panels render bg-black on top, so aurora doesn't bleed through video.
export function AuroraBg({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        "fixed inset-0 -z-10 overflow-hidden bg-ink-950 pointer-events-none",
        className
      )}
    >
      <div
        className={cn(
          `
          [--white-gradient:repeating-linear-gradient(100deg,var(--white)_0%,var(--white)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--white)_16%)]
          [--dark-gradient:repeating-linear-gradient(100deg,var(--black)_0%,var(--black)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--black)_16%)]
          [--aurora:repeating-linear-gradient(100deg,var(--blue-500)_10%,var(--indigo-300)_15%,var(--blue-300)_20%,var(--violet-200)_25%,var(--blue-400)_30%)]
          [background-image:var(--dark-gradient),var(--aurora)]
          [background-size:300%,_200%]
          [background-position:50%_50%,50%_50%]
          filter blur-[14px]
          after:content-[""] after:absolute after:inset-0 after:[background-image:var(--dark-gradient),var(--aurora)]
          after:[background-size:200%,_100%]
          after:animate-aurora after:[background-attachment:fixed] after:mix-blend-difference
          absolute -inset-[10px] opacity-[0.18] will-change-transform
          [mask-image:radial-gradient(ellipse_at_50%_30%,black_25%,var(--transparent)_85%)]
          `
        )}
      />
    </div>
  );
}
