import { useMemo } from "react";
import { useAnalysisStore } from "../state/analysis";
import type { Residual } from "../lib/api";

const WIDTH = 400;
const HEIGHT = 90;
const PAD_TOP = 6;
const PAD_BOTTOM = 6;

// Zustand's default equality is reference-based; returning `?? []` inside the
// selector allocates a fresh array on every call and blows up with "Maximum
// update depth exceeded". Select the stable field and default outside.
const EMPTY_RESIDUALS: Residual[] = [];

export function ResidualChart() {
  const analysisResult = useAnalysisStore((s) => s.analysisResult);
  const residuals = analysisResult?.residuals ?? EMPTY_RESIDUALS;

  const { path, peakSigma, threeSigmaY } = useMemo(() => {
    if (residuals.length < 2) {
      return { path: null, peakSigma: 0, threeSigmaY: HEIGHT };
    }
    const maxSigma = Math.max(...residuals.map((r) => r.sigma), 3);
    const maxT = Math.max(...residuals.map((r) => r.t_s), 0.001);
    const usableH = HEIGHT - PAD_TOP - PAD_BOTTOM;

    const toY = (sigma: number) =>
      HEIGHT - PAD_BOTTOM - (Math.min(sigma, maxSigma) / maxSigma) * usableH;

    const d = residuals
      .map((r, i) => {
        const x = (r.t_s / maxT) * WIDTH;
        const y = toY(r.sigma);
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");

    return {
      path: d,
      peakSigma: Math.max(...residuals.map((r) => r.sigma)),
      threeSigmaY: toY(3),
    };
  }, [residuals]);

  const flag = peakSigma > 5 ? "flagged" : peakSigma > 3 ? "borderline" : "clean";
  const flagColor =
    peakSigma > 5
      ? "text-red-400"
      : peakSigma > 3
        ? "text-amber-400"
        : "text-emerald-400";

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="font-mono text-[10px] uppercase tracking-widest text-neutral-500">
          physics residual · σ over time
        </div>
        <div className="font-mono text-[11px] text-neutral-400">
          peak <span className={flagColor}>{peakSigma.toFixed(2)}σ</span>{" "}
          <span className="text-neutral-600">·</span>{" "}
          <span className={flagColor}>{flag}</span>
        </div>
      </div>
      {path ? (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-24 w-full"
          preserveAspectRatio="none"
        >
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
          <path
            d={path}
            fill="none"
            stroke="#ef4444"
            strokeWidth="1.6"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      ) : (
        <div className="flex h-24 items-center justify-center rounded-md border border-dashed border-neutral-800 bg-neutral-950/40 text-xs text-neutral-500">
          upload a video to compute residuals
        </div>
      )}
    </div>
  );
}
