import { useMemo } from "react";
import { useAnalysisStore } from "../state/analysis";
import type { ObjectReport } from "../lib/api";

const VERDICT_STYLE: Record<string, { border: string; chip: string }> = {
  consistent: {
    border: "border-emerald-400/80",
    chip: "bg-emerald-400/90 text-emerald-950",
  },
  borderline: {
    border: "border-amber-400/80",
    chip: "bg-amber-400/90 text-amber-950",
  },
  implausible: {
    border: "border-red-500/90",
    chip: "bg-red-500/90 text-white",
  },
  morphing: {
    border: "border-fuchsia-500/90",
    chip: "bg-fuchsia-500/90 text-white",
  },
  agent: {
    border: "border-sky-400/60",
    chip: "bg-sky-400/80 text-sky-950",
  },
  static: {
    border: "border-neutral-500/50",
    chip: "bg-neutral-500/80 text-neutral-100",
  },
};

/**
 * Labeled bounding boxes over the video, following the playhead.
 *
 * Boxes come normalized to [0,1] so they position with CSS percentages —
 * no need to know the pixel size of the rendered <video>. Color encodes
 * the per-object verdict; the label chip names what YOLO saw and, for
 * flagged objects, the σ / morph score that condemned it.
 */
export function VideoOverlay() {
  const objects = useAnalysisStore((s) => s.analysisResult?.objects);
  const fps = useAnalysisStore((s) => s.analysisResult?.fps ?? 30);
  const currentTimeS = useAnalysisStore((s) => s.currentTimeS);

  const frameIdx = Math.max(0, Math.round(currentTimeS * (fps || 30)));

  const visible = useMemo(() => {
    if (!objects) return [];
    const out: { obj: ObjectReport; box: [number, number, number, number] }[] =
      [];
    for (const obj of objects) {
      // Exact frame, else nearest within 2 frames (detector can blip).
      let box = obj.boxes_norm[String(frameIdx)];
      if (!box) {
        for (const d of [1, -1, 2, -2]) {
          box = obj.boxes_norm[String(frameIdx + d)];
          if (box) break;
        }
      }
      if (box) out.push({ obj, box });
    }
    return out;
  }, [objects, frameIdx]);

  if (visible.length === 0) return null;

  return (
    <div className="pointer-events-none absolute inset-0">
      {visible.map(({ obj, box }) => {
        const style = VERDICT_STYLE[obj.verdict] ?? VERDICT_STYLE.static;
        const [x0, y0, x1, y1] = box;
        const detail =
          obj.verdict === "morphing"
            ? `morph ${(obj.morph_score * 100).toFixed(0)}%`
            : obj.ballistic.eligible
              ? `${obj.ballistic.sigma.toFixed(1)}σ`
              : obj.verdict;
        return (
          <div
            key={obj.object_id}
            className={`absolute rounded border-2 ${style.border}`}
            style={{
              left: `${x0 * 100}%`,
              top: `${y0 * 100}%`,
              width: `${(x1 - x0) * 100}%`,
              height: `${(y1 - y0) * 100}%`,
            }}
          >
            <div
              className={`absolute -top-5 left-0 whitespace-nowrap rounded px-1.5 py-0.5 font-mono text-[10px] leading-none ${style.chip}`}
            >
              {obj.label} · {detail}
            </div>
          </div>
        );
      })}
    </div>
  );
}
