import { useCallback, useEffect, useRef, useState } from "react";
import { useAnalysisStore } from "../state/analysis";
import { VideoOverlay } from "./VideoOverlay";

type RVFCVideoElement = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    cb: (now: number, metadata: { mediaTime: number }) => void,
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

function fmt(t: number): string {
  if (!Number.isFinite(t)) return "0.00";
  return t.toFixed(2);
}

export function Player() {
  const videoUrl = useAnalysisStore((s) => s.videoUrl);
  const reset = useAnalysisStore((s) => s.reset);
  const seekRequestS = useAnalysisStore((s) => s.seekRequestS);
  const requestSeek = useAnalysisStore((s) => s.requestSeek);
  const setCurrentTimeS = useAnalysisStore((s) => s.setCurrentTimeS);
  const setDurationS = useAnalysisStore((s) => s.setDurationS);
  const currentTimeS = useAnalysisStore((s) => s.currentTimeS);
  const durationS = useAnalysisStore((s) => s.durationS);
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);

  const toggle = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    if (el.paused) {
      void el.play();
    } else {
      el.pause();
    }
  }, []);

  // Watch external seek requests (from Timeline scrubber).
  useEffect(() => {
    if (seekRequestS === null) return;
    const el = ref.current;
    if (!el) return;
    el.currentTime = Math.max(0, Math.min(durationS || seekRequestS, seekRequestS));
    requestSeek(null);
  }, [seekRequestS, durationS, requestSeek]);

  useEffect(() => {
    const el = ref.current as RVFCVideoElement | null;
    if (!el) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setCurrentTimeS(el.currentTime);
    const onMeta = () => setDurationS(el.duration);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("seeked", onTime);

    let handle: number | undefined;
    let cancelled = false;
    const rvfc = el.requestVideoFrameCallback?.bind(el);
    const cancelRvfc = el.cancelVideoFrameCallback?.bind(el);
    const step = (_now: number, metadata: { mediaTime: number }) => {
      if (cancelled) return;
      const fps = 30;
      setFrame(Math.round(metadata.mediaTime * fps));
      // Sub-timeupdate resolution — push RVFC's mediaTime for smoother 3D sync.
      setCurrentTimeS(metadata.mediaTime);
      if (rvfc) handle = rvfc(step);
    };
    if (rvfc) handle = rvfc(step);

    return () => {
      cancelled = true;
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("seeked", onTime);
      if (handle !== undefined && cancelRvfc) cancelRvfc(handle);
    };
  }, [videoUrl, setCurrentTimeS, setDurationS]);

  if (!videoUrl) return null;

  return (
    <div className="flex h-full flex-col gap-3 rounded-lg border border-neutral-800 bg-neutral-950 p-3">
      <div className="relative overflow-hidden rounded-md bg-black">
        <video
          ref={ref}
          src={videoUrl}
          className="h-full w-full"
          playsInline
        />
        <VideoOverlay />
      </div>
      <div className="flex items-center gap-4 font-mono text-xs text-neutral-400">
        <button
          type="button"
          onClick={toggle}
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-1 text-neutral-200 transition-colors hover:border-neutral-500 hover:bg-neutral-800"
        >
          {playing ? "pause" : "play"}
        </button>
        <div className="tabular-nums">
          {fmt(currentTimeS)}s / {fmt(durationS)}s
        </div>
        <div className="tabular-nums">frame {frame}</div>
        <button
          type="button"
          onClick={reset}
          className="ml-auto rounded border border-neutral-800 px-2 py-1 text-neutral-500 transition-colors hover:border-neutral-600 hover:text-neutral-300"
        >
          clear
        </button>
      </div>
    </div>
  );
}
