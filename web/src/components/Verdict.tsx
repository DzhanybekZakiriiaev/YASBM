import { useAnalysisStore } from "../state/analysis";

type Tone = "neutral" | "good" | "bad" | "warn";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "border-neutral-800 text-neutral-300",
  good: "border-emerald-900/60 bg-emerald-950/20 text-emerald-200",
  warn: "border-amber-900/60 bg-amber-950/20 text-amber-200",
  bad: "border-red-900/60 bg-red-950/20 text-red-200",
};

export function Verdict() {
  const status = useAnalysisStore((s) => s.status);
  const analysisResult = useAnalysisStore((s) => s.analysisResult);
  const verdict = useAnalysisStore((s) => s.verdict);
  const error = useAnalysisStore((s) => s.error);
  const videoUrl = useAnalysisStore((s) => s.videoUrl);

  let title: string;
  let body: string;
  let tone: Tone = "neutral";

  if (error) {
    title = "error";
    body = error;
    tone = "bad";
  } else if (!videoUrl) {
    title = "awaiting upload";
    body =
      "drop a video clip in the panel to the left. 2–6 second clips with a rigid moving object work best.";
  } else if (status === "uploading") {
    title = "uploading";
    body = "sending clip to pipeline…";
  } else if (status === "analyzing") {
    title = "analyzing";
    body =
      "segment → track → depth → scene → lift → physics — this takes a few seconds.";
  } else if (analysisResult) {
    const sigma = analysisResult.verdict_score;
    if (sigma < 3) {
      title = "consistent with real physics";
      tone = "good";
    } else if (sigma < 10) {
      title = "borderline";
      tone = "warn";
    } else {
      title = "physically implausible";
      tone = "bad";
    }
    // If Claude has started streaming a verdict, prefer that as the body.
    // Otherwise fall back to a deterministic sigma-based summary so the
    // card still reads well without the LLM (e.g. no Anthropic key set).
    if (verdict) {
      body = verdict;
    } else if (sigma < 3) {
      body = `peak residual ${sigma.toFixed(2)} σ. Observed motion matches a Newtonian trajectory within measurement noise.`;
    } else if (sigma < 10) {
      body = `peak residual ${sigma.toFixed(1)} σ. Some frames diverge from a Newtonian fit — inspect the trajectory in the 3D viewer.`;
    } else {
      body = `peak residual ${sigma.toFixed(1)} σ, well beyond the empirical noise floor. The observed trajectory is inconsistent with any Newtonian projectile that could have produced the release conditions.`;
    }
  } else {
    title = "awaiting analysis";
    body = "the pipeline should fire automatically after upload.";
  }

  const cls = TONE_CLASS[tone];

  return (
    <div className={`rounded-lg border bg-neutral-950 p-4 ${cls}`}>
      <div className="mb-2 flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-widest opacity-70">
          verdict
        </div>
        <div className="font-mono text-[10px] uppercase tracking-widest opacity-50">
          {status}
        </div>
      </div>
      <div className="mb-1 text-base font-medium">{title}</div>
      <div className="text-sm leading-relaxed opacity-80">{body}</div>
    </div>
  );
}
