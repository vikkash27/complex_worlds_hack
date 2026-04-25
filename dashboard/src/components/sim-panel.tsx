import { useRef, useState } from "react";

const VIDEO_SRC = "/isaac-sim.mp4";

export function SimPanel({ stage }: { stage: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [hasVideo, setHasVideo] = useState(true);

  return (
    <div className="absolute inset-0 bg-black">
      {hasVideo && (
        <video
          ref={ref}
          src={VIDEO_SRC}
          autoPlay
          loop
          muted
          playsInline
          onError={() => setHasVideo(false)}
          className="w-full h-full object-cover"
        />
      )}
      {!hasVideo && (
        <div className="absolute inset-0 flex items-center justify-center bg-[radial-gradient(ellipse_at_30%_25%,rgba(124,242,196,0.06),transparent_55%),radial-gradient(ellipse_at_75%_85%,rgba(90,200,250,0.05),transparent_55%)]">
          <div className="text-center">
            <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-neutral-600 mb-2">
              Isaac Sim
            </div>
            <div className="text-neutral-400 text-sm font-sans">
              drop <code className="text-emerald-300 font-mono text-xs">public/isaac-sim.mp4</code>
            </div>
          </div>
        </div>
      )}

      {/* Single subtle overlay — current stage badge bottom-left */}
      <div className="absolute bottom-3 left-3 flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-black/55 backdrop-blur border border-white/10">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-neutral-300">
          stage
        </span>
        <span className="text-[11px] font-medium text-emerald-300 tracking-tight">
          {stage}
        </span>
      </div>
    </div>
  );
}
