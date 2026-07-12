import { create } from "zustand";
import * as api from "../lib/api";

export type AnalysisStatus =
  | "idle"
  | "uploading"
  | "analyzing"
  | "done"
  | "error";

export interface AnalysisProgress {
  stage: string;
  pct: number;
}

export type FlythroughStatus = "idle" | "generating" | "done" | "error";

interface AnalysisState {
  videoFile: File | null;
  videoUrl: string | null;
  status: AnalysisStatus;
  progress: AnalysisProgress | null;
  verdict: string | null;
  analysisResult: api.AnalyzeResponse | null;
  error: string | null;
  // ViewCrafter flythrough state
  flythroughStatus: FlythroughStatus;
  flythroughUrl: string | null;
  flythroughError: string | null;
  // Playhead — video's current time in seconds. Player pushes this in on
  // RVFC / timeupdate; Timeline reads it to render the scrubber head;
  // Viewer3D reads it to animate per-frame track markers.
  currentTimeS: number;
  durationS: number;
  // Seek request: external components (Timeline) set this; Player watches
  // and seeks the underlying <video>, then clears back to null.
  seekRequestS: number | null;
  setVideo: (file: File) => void;
  reset: () => void;
  setStatus: (status: AnalysisStatus) => void;
  setProgress: (progress: AnalysisProgress | null) => void;
  setVerdict: (verdict: string | null) => void;
  setError: (error: string | null) => void;
  runAnalyze: () => Promise<void>;
  runFlythrough: (frame: Blob) => Promise<void>;
  clearFlythrough: () => void;
  setCurrentTimeS: (t: number) => void;
  setDurationS: (t: number) => void;
  requestSeek: (t: number | null) => void;
}

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  videoFile: null,
  videoUrl: null,
  status: "idle",
  progress: null,
  verdict: null,
  analysisResult: null,
  error: null,
  flythroughStatus: "idle",
  flythroughUrl: null,
  flythroughError: null,
  currentTimeS: 0,
  durationS: 0,
  seekRequestS: null,
  setVideo: (file) => {
    const prev = get().videoUrl;
    if (prev) URL.revokeObjectURL(prev);
    const url = URL.createObjectURL(file);
    set({
      videoFile: file,
      videoUrl: url,
      status: "idle",
      progress: null,
      verdict: null,
      analysisResult: null,
      error: null,
      flythroughStatus: "idle",
      flythroughUrl: null,
      flythroughError: null,
      currentTimeS: 0,
      durationS: 0,
      seekRequestS: null,
    });
  },
  reset: () => {
    const prev = get().videoUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      videoFile: null,
      videoUrl: null,
      status: "idle",
      progress: null,
      verdict: null,
      analysisResult: null,
      error: null,
      flythroughStatus: "idle",
      flythroughUrl: null,
      flythroughError: null,
      currentTimeS: 0,
      durationS: 0,
      seekRequestS: null,
    });
  },
  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress }),
  setVerdict: (verdict) => set({ verdict }),
  setError: (error) => set({ error }),
  runAnalyze: async () => {
    const file = get().videoFile;
    if (!file) return;
    set({
      status: "analyzing",
      error: null,
      analysisResult: null,
      verdict: null,
    });
    try {
      const result = await api.analyze(file);
      set({
        status: "done",
        analysisResult: result,
        error: null,
      });

      // Chain Claude verdict streaming — best-effort. A verdict-endpoint
      // failure (no Anthropic secret configured, model rate-limit, etc.)
      // shouldn't invalidate the analyze result, so we swallow the error
      // and leave the fallback score-based verdict copy in place.
      try {
        let accumulated = "";
        await api.verdictStream(
          {
            verdict_score: result.verdict_score,
            residuals: result.residuals,
            clip_duration_s: result.duration_s ?? undefined,
            verdict_basis: result.verdict_basis ?? undefined,
            objects: (result.objects ?? []).map((o) => ({
              label: o.label,
              verdict: o.verdict,
              sigma: o.ballistic.sigma,
              morph_score: o.morph_score,
              reason: o.ballistic.reason,
            })),
          },
          (chunk) => {
            accumulated += chunk;
            set({ verdict: accumulated });
          },
        );
      } catch (verdictErr) {
        // eslint-disable-next-line no-console
        console.warn("verdict stream failed:", verdictErr);
      }
    } catch (err) {
      set({
        status: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
  runFlythrough: async (frame: Blob) => {
    set({
      flythroughStatus: "generating",
      flythroughUrl: null,
      flythroughError: null,
    });
    try {
      const result = await api.generateFlythrough(frame);
      set({
        flythroughStatus: "done",
        flythroughUrl: result.flythrough_url,
        flythroughError: null,
      });
    } catch (err) {
      set({
        flythroughStatus: "error",
        flythroughError: err instanceof Error ? err.message : String(err),
      });
    }
  },
  clearFlythrough: () => {
    set({
      flythroughStatus: "idle",
      flythroughUrl: null,
      flythroughError: null,
    });
  },
  setCurrentTimeS: (t) => set({ currentTimeS: t }),
  setDurationS: (t) => set({ durationS: t }),
  requestSeek: (t) => set({ seekRequestS: t }),
}));
