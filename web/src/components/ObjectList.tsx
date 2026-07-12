import { useAnalysisStore } from "../state/analysis";

const VERDICT_CHIP: Record<string, string> = {
  consistent: "border-emerald-900/60 bg-emerald-950/30 text-emerald-300",
  borderline: "border-amber-900/60 bg-amber-950/30 text-amber-300",
  implausible: "border-red-900/60 bg-red-950/30 text-red-300",
  morphing: "border-fuchsia-900/60 bg-fuchsia-950/30 text-fuchsia-300",
  agent: "border-sky-900/60 bg-sky-950/30 text-sky-300",
  static: "border-neutral-800 bg-neutral-950 text-neutral-400",
};

const VERDICT_HINT: Record<string, string> = {
  consistent: "free motion matches Newtonian fit",
  borderline: "some divergence from ballistic fit",
  implausible: "trajectory violates projectile physics",
  morphing: "identity / rigid shape unstable over time",
  agent: "self-propelled — ballistic check not applicable",
  static: "no significant motion to audit",
};

/** Per-object verdict chips under the main verdict card. */
export function ObjectList() {
  const objects = useAnalysisStore((s) => s.analysisResult?.objects);

  if (!objects || objects.length === 0) return null;

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
      <div className="mb-3 font-mono text-[10px] uppercase tracking-widest text-neutral-500">
        detected objects · {objects.length}
      </div>
      <div className="flex flex-wrap gap-2">
        {objects.map((obj) => {
          const cls = VERDICT_CHIP[obj.verdict] ?? VERDICT_CHIP.static;
          const detail = obj.ballistic.eligible
            ? `${obj.ballistic.sigma.toFixed(1)}σ`
            : obj.verdict === "morphing"
              ? `morph ${(obj.morph_score * 100).toFixed(0)}%`
              : obj.verdict;
          return (
            <div
              key={obj.object_id}
              title={`${VERDICT_HINT[obj.verdict] ?? ""} · flicker ${(obj.class_flicker * 100).toFixed(0)}% · rigidity cv ${(obj.rigidity_cv * 100).toFixed(1)}% · ${obj.frames_present} frames`}
              className={`cursor-default rounded-md border px-2.5 py-1.5 font-mono text-[11px] ${cls}`}
            >
              <span className="font-semibold">{obj.label}</span>
              <span className="mx-1.5 opacity-40">·</span>
              {detail}
            </div>
          );
        })}
      </div>
    </div>
  );
}
