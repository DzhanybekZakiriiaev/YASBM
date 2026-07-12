import { useAnalysisStore, type HeroStatus } from "../state/analysis";

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
  const props = useAnalysisStore((s) => s.analysisResult?.props);
  const heroStatus = useAnalysisStore((s) => s.heroStatus);
  const runHero = useAnalysisStore((s) => s.runHero);

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
          // Hero (Tripo image-to-3D) is only offered when the pipeline saved
          // a crop of the object to feed the generator.
          const hasCrop = props?.some(
            (p) => p.object_id === obj.object_id && p.crop_url,
          );
          return (
            <div
              key={obj.object_id}
              title={`${VERDICT_HINT[obj.verdict] ?? ""} · flicker ${(obj.class_flicker * 100).toFixed(0)}% · rigidity cv ${(obj.rigidity_cv * 100).toFixed(1)}% · ${obj.frames_present} frames`}
              className={`cursor-default rounded-md border px-2.5 py-1.5 font-mono text-[11px] ${cls}`}
            >
              <span className="font-semibold">{obj.label}</span>
              <span className="mx-1.5 opacity-40">·</span>
              {detail}
              {hasCrop ? (
                <>
                  <span className="mx-1.5 opacity-40">·</span>
                  <HeroScanButton
                    status={heroStatus[obj.object_id]}
                    onClick={() => void runHero(obj.object_id)}
                  />
                </>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const HERO_LABEL: Record<HeroStatus, string> = {
  generating: "scanning…",
  done: "scanned ✓",
  error: "retry",
  unavailable: "unavailable",
};

/**
 * Tiny text button that kicks off Tripo hero-asset generation for one
 * object. "unavailable" = backend returned 503 (Tripo not configured) —
 * kept visible but inert so the state is legible rather than vanishing.
 */
function HeroScanButton({
  status,
  onClick,
}: {
  status: HeroStatus | undefined;
  onClick: () => void;
}) {
  const label = status ? HERO_LABEL[status] : "3d scan";
  const disabled =
    status === "generating" || status === "done" || status === "unavailable";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="font-mono text-[10px] uppercase tracking-wider underline decoration-dotted underline-offset-2 opacity-70 transition-opacity hover:opacity-100 disabled:cursor-default disabled:no-underline disabled:opacity-50"
    >
      {label}
    </button>
  );
}
