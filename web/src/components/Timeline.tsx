import { useCallback, useMemo, useRef, useState } from "react";
import { useAnalysisStore } from "../state/analysis";
import type { Residual } from "../lib/api";

const WIDTH = 800;
const HEIGHT = 120;
const PAD_TOP = 8;
const PAD_BOTTOM = 8;

/**
 * Interactive scrubbable timeline.
 *
 * X axis = video time. Y axis = σ. The observed residual curve is drawn as a
 * red path; a dashed amber line marks the 3σ threshold. A vertical white
 * playhead tracks the video's currentTimeS from the store. Clicking or
 * dragging anywhere on the strip issues a seek request, which the Player
 * component watches and applies to the underlying <video>.
 *
 * On hover, a tooltip shows the exact frame time + σ + Δm at that point.
 */
const EMPTY_RESIDUALS: Residual[] = [];

export function Timeline() {
  const residuals = useAnalysisStore(
    (s) => s.analysisResult?.residuals ?? EMPTY_RESIDUALS,
  );
  const currentTimeS = useAnalysisStore((s) => s.currentTimeS);
  const durationS = useAnalysisStore((s) => s.durationS);
  const requestSeek = useAnalysisStore((s) => s.requestSeek);
  const [hover, setHover] = useState<{
    x: number;
    residual: Residual;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const { path, maxSigma, maxT, threeSigmaY, toY } = useMemo(() => {
    if (residuals.length < 2) {
      return {
        path: null,
        maxSigma: 3,
        maxT: durationS || 1,
        threeSigmaY: HEIGHT,
        toY: (_: number) => HEIGHT,
      };
    }
    const mSigma = Math.max(3, ...residuals.map((r) => r.sigma));
    const mT = Math.max(durationS, ...residuals.map((r) => r.t_s), 0.001);
    const usable = HEIGHT - PAD_TOP - PAD_BOTTOM;
    const yFn = (sigma: number) =>
      HEIGHT - PAD_BOTTOM - (Math.min(sigma, mSigma) / mSigma) * usable;
    const d = residuals
      .map((r, i) => {
        const x = (r.t_s / mT) * WIDTH;
        const y = yFn(r.sigma);
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
    return {
      path: d,
      maxSigma: mSigma,
      maxT: mT,
      threeSigmaY: yFn(3),
      toY: yFn,
    };
  }, [residuals, durationS]);

  const seekFromEvent = useCallback(
    (evt: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const t = (x / rect.width) * maxT;
      requestSeek(Math.max(0, Math.min(maxT, t)));
    },
    [maxT, requestSeek],
  );

  const onHover = useCallback(
    (evt: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg || residuals.length === 0) return;
      const rect = svg.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const svgX = (x / rect.width) * WIDTH;
      const t = (x / rect.width) * maxT;
      // Nearest residual by time.
      let nearest = residuals[0];
      let bestDelta = Math.abs(nearest.t_s - t);
      for (const r of residuals) {
        const d = Math.abs(r.t_s - t);
        if (d < bestDelta) {
          bestDelta = d;
          nearest = r;
        }
      }
      setHover({ x: svgX, residual: nearest });
      if (dragging) seekFromEvent(evt);
    },
    [residuals, maxT, dragging, seekFromEvent],
  );

  const playheadX =
    maxT > 0 ? (Math.min(currentTimeS, maxT) / maxT) * WIDTH : 0;
  const peakSigma = residuals.reduce((m, r) => Math.max(m, r.sigma), 0);
  const peakFlag =
    peakSigma > 10 ? "flagged" : peakSigma > 3 ? "borderline" : "clean";
  const peakColor =
    peakSigma > 10
      ? "text-red-400"
      : peakSigma > 3
        ? "text-amber-400"
        : "text-emerald-400";

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="font-mono text-[10px] uppercase tracking-widest text-neutral-500">
          physics timeline · σ vs. time · click to seek
        </div>
        <div className="font-mono text-[11px] text-neutral-400">
          peak <span className={peakColor}>{peakSigma.toFixed(2)}σ</span>{" "}
          <span className="text-neutral-600">·</span>{" "}
          <span className={peakColor}>{peakFlag}</span>
        </div>
      </div>
      {path ? (
        <div className="relative select-none">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="h-28 w-full cursor-crosshair"
            preserveAspectRatio="none"
            onMouseDown={(e) => {
              setDragging(true);
              seekFromEvent(e);
            }}
            onMouseUp={() => setDragging(false)}
            onMouseLeave={() => {
              setDragging(false);
              setHover(null);
            }}
            onMouseMove={onHover}
          >
            {/* 3σ threshold */}
            <line
              x1="0"
              y1={threeSigmaY}
              x2={WIDTH}
              y2={threeSigmaY}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth="0.8"
              opacity="0.5"
            />
            {/* σ curve */}
            <path
              d={path}
              fill="none"
              stroke="#ef4444"
              strokeWidth="1.6"
              vectorEffect="non-scaling-stroke"
            />
            {/* hover highlight */}
            {hover && (
              <>
                <circle
                  cx={hover.x}
                  cy={toY(hover.residual.sigma)}
                  r="3.5"
                  fill="#fff"
                  stroke="#ef4444"
                  strokeWidth="1"
                  vectorEffect="non-scaling-stroke"
                />
                <line
                  x1={hover.x}
                  y1="0"
                  x2={hover.x}
                  y2={HEIGHT}
                  stroke="#ffffff"
                  strokeOpacity="0.15"
                  strokeWidth="0.6"
                  vectorEffect="non-scaling-stroke"
                />
              </>
            )}
            {/* playhead */}
            <line
              x1={playheadX}
              y1="0"
              x2={playheadX}
              y2={HEIGHT}
              stroke="#e5e5e5"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
              opacity="0.9"
            />
            <circle
              cx={playheadX}
              cy="8"
              r="4"
              fill="#e5e5e5"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
          {hover && (
            <HoverTooltip
              residual={hover.residual}
              containerX={hover.x / WIDTH}
              maxSigma={maxSigma}
            />
          )}
        </div>
      ) : (
        <div className="flex h-28 items-center justify-center rounded-md border border-dashed border-neutral-800 bg-neutral-950/40 text-xs text-neutral-500">
          upload a video to compute residuals
        </div>
      )}
    </div>
  );
}

function HoverTooltip({
  residual,
  containerX,
  maxSigma,
}: {
  residual: Residual;
  containerX: number;
  maxSigma: number;
}) {
  const verdict =
    residual.sigma > 10
      ? "physically implausible"
      : residual.sigma > 3
        ? "borderline"
        : "consistent";
  const color =
    residual.sigma > 10
      ? "text-red-300"
      : residual.sigma > 3
        ? "text-amber-300"
        : "text-emerald-300";

  // Position the tooltip; flip to the left side when the cursor is past ~70%.
  const leftPct = containerX * 100;
  const rightSide = containerX > 0.7;

  return (
    <div
      className={`pointer-events-none absolute top-2 z-10 min-w-[180px] rounded-md border border-neutral-700 bg-neutral-950/95 p-2.5 font-mono text-[10px] text-neutral-300 shadow-lg`}
      style={{
        left: rightSide ? undefined : `${leftPct}%`,
        right: rightSide ? `${100 - leftPct}%` : undefined,
        transform: rightSide ? "translateX(-12px)" : "translateX(12px)",
      }}
    >
      <div className="mb-1 text-neutral-500">
        t = {residual.t_s.toFixed(3)}s
      </div>
      <div>
        σ ={" "}
        <span className={color}>
          {residual.sigma.toFixed(2)} / {maxSigma.toFixed(1)}
        </span>
      </div>
      <div>
        Δ = <span className="text-neutral-200">{residual.delta_m.toFixed(4)} m</span>
      </div>
      <div className={`mt-1 uppercase tracking-widest ${color}`}>{verdict}</div>
    </div>
  );
}
