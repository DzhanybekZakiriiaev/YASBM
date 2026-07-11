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

interface AnalysisState {
  videoFile: File | null;
  videoUrl: string | null;
  status: AnalysisStatus;
  progress: AnalysisProgress | null;
  verdict: string | null;
  analysisResult: api.AnalyzeResponse | null;
  error: string | null;
  setVideo: (file: File) => void;
  reset: () => void;
  setStatus: (status: AnalysisStatus) => void;
  setProgress: (progress: AnalysisProgress | null) => void;
  setVerdict: (verdict: string | null) => void;
  setError: (error: string | null) => void;
  runAnalyze: () => Promise<void>;
}

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  videoFile: null,
  videoUrl: null,
  status: "idle",
  progress: null,
  verdict: null,
  analysisResult: null,
  error: null,
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
}));
