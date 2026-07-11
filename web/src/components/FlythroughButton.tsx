import { useCallback, useMemo } from "react";
import { useAnalysisStore } from "../state/analysis";
import { VIEWCRAFTER_URL } from "../lib/api";

/**
 * "Generate 360° flythrough" button.
 *
 * Grabs the first frame of the loaded video as a PNG, POSTs it to the
 * ViewCrafter Modal deployment, then shows the returned MP4 in an inline
 * player. Hidden when VITE_VIEWCRAFTER_URL is not configured.
 *
 * Note: this is *AI-generated content* on top of the observed reconstruction.
 * Clearly labelled — the badge on the resulting video reads
 * "AI-COMPLETED VIEW" so a forensic user is never confused about which
 * pixels are observed and which are hallucinated.
 */
export function FlythroughButton() {
  const videoUrl = useAnalysisStore((s) => s.videoUrl);
  const flythroughStatus = useAnalysisStore((s) => s.flythroughStatus);
  const flythroughUrl = useAnalysisStore((s) => s.flythroughUrl);
  const flythroughError = useAnalysisStore((s) => s.flythroughError);
  const runFlythrough = useAnalysisStore((s) => s.runFlythrough);
  const clearFlythrough = useAnalysisStore((s) => s.clearFlythrough);

  const disabled = useMemo(
    () =>
      !VIEWCRAFTER_URL ||
      !videoUrl ||
      flythroughStatus === "generating",
    [videoUrl, flythroughStatus],
  );

  const onClick = useCallback(async () => {
    if (!videoUrl) return;

    // Load the video off-screen so we can grab a still.
    const video = document.createElement("video");
    video.src = videoUrl;
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.playsInline = true;

    await new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error("failed to load video frame"));
    });

    video.currentTime = 0;
    await new Promise<void>((resolve) => {
      const done = () => {
        video.removeEventListener("seeked", done);
        resolve();
      };
      video.addEventListener("seeked", done);
    });

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((b) => resolve(b), "image/png");
    });
    if (!blob) return;

    await runFlythrough(blob);
  }, [videoUrl, runFlythrough]);

  if (!VIEWCRAFTER_URL) return null;

  return (
    <>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-neutral-200 transition-colors hover:border-neutral-700 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:border-neutral-900 disabled:bg-neutral-950 disabled:text-neutral-600"
      >
        {flythroughStatus === "generating"
          ? "generating…"
          : "ai 360° flythrough"}
      </button>
      {flythroughStatus === "generating" && (
        <div className="mt-2 rounded-md border border-neutral-800 bg-neutral-950 p-3 text-xs text-neutral-400">
          ViewCrafter is generating a 360° flythrough. This takes ~60–120 s on
          a warm container, up to a few minutes on the first cold start.
        </div>
      )}
      {flythroughError && (
        <div className="mt-2 rounded-md border border-red-900/60 bg-red-950/20 p-3 text-xs text-red-300">
          {flythroughError}
        </div>
      )}
      {flythroughStatus === "done" && flythroughUrl && (
        <FlythroughOverlay
          url={flythroughUrl}
          onClose={clearFlythrough}
        />
      )}
    </>
  );
}

function FlythroughOverlay({
  url,
  onClose,
}: {
  url: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-6">
      <div className="w-full max-w-3xl rounded-lg border border-neutral-800 bg-neutral-950 p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-baseline gap-3">
            <div className="font-mono text-[10px] uppercase tracking-widest text-amber-400">
              ai-completed view
            </div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-neutral-500">
              viewcrafter
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-neutral-300 hover:border-neutral-700 hover:bg-neutral-800"
          >
            close
          </button>
        </div>
        <video
          src={url}
          controls
          autoPlay
          loop
          muted
          className="w-full rounded-md border border-neutral-900"
        />
        <div className="mt-3 text-xs leading-relaxed text-neutral-400">
          These frames are hallucinated by a video-diffusion model
          (ViewCrafter, TPAMI 2025) from the observed first frame + recovered
          point cloud. They are not physical evidence — they show what an AI
          thinks the room might look like when viewed from angles the phone
          never saw. Use for context only, never as forensic evidence.
        </div>
      </div>
    </div>
  );
}
