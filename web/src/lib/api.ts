// API base URL preference order:
//   1. VITE_EDGE_URL       — Vercel/Hono edge layer (production)
//   2. VITE_PIPELINE_URL   — direct Modal pipeline URL (staging / manual)
//   3. http://127.0.0.1:8001 — local pipeline dev server (fallback)
//
// The edge layer will eventually front all pipeline traffic (auth, rate limit,
// R2 signing, SSE relay). Until it is deployed we still allow the client to
// hit the pipeline directly by setting VITE_PIPELINE_URL.
export const API_BASE_URL: string =
  (import.meta.env.VITE_EDGE_URL as string | undefined) ??
  (import.meta.env.VITE_PIPELINE_URL as string | undefined) ??
  "http://127.0.0.1:8001";

// Backwards-compat alias — some modules still import PIPELINE_URL.
export const PIPELINE_URL: string = API_BASE_URL;

// Separate deployment for ViewCrafter (yasbm-viewcrafter Modal app).
// Optional — the button is hidden if this env var is not set.
export const VIEWCRAFTER_URL: string | null =
  (import.meta.env.VITE_VIEWCRAFTER_URL as string | undefined) ?? null;

export type Vec3 = [number, number, number];

export interface TrajectoryPoint {
  t_s: number;
  position: Vec3;
}

export interface Track {
  track_id: number;
  label: string;
  points: TrajectoryPoint[];
  /** Normalized (u, v) pixel positions in [0, 1] per frame. */
  points_2d?: [number, number][] | null;
  /** Per-frame σ for this specific track (aligned with `points`). */
  sigma_per_frame?: number[] | null;
}

export interface Residual {
  t_s: number;
  delta_m: number;
  sigma: number;
}

export interface AnalyzeResponse {
  status: string;
  tracks: Track[];
  residuals: Residual[];
  verdict_score: number;
  point_cloud_url: string | null;
  error?: string | null;
}

export async function health(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return (await res.json()) as { status: string };
}

export interface VerdictRequest {
  verdict_score: number;
  residuals: Residual[];
  clip_duration_s?: number;
}

/**
 * Stream Claude's verdict card via SSE. Calls `onToken` for each text
 * chunk, resolves on `{type: "done"}`, rejects on `{type: "error"}` or
 * transport failure. Abortable via `signal`.
 */
export async function verdictStream(
  req: VerdictRequest,
  onToken: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/verdict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    let msg = `verdict failed: ${res.status}`;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j.detail) msg = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!raw.startsWith("data:")) continue;
      const payload = raw.slice(5).trim();
      try {
        const msg = JSON.parse(payload) as
          | { type: "token"; text: string }
          | { type: "done" }
          | { type: "error"; message: string };
        if (msg.type === "token") onToken(msg.text);
        else if (msg.type === "done") return;
        else if (msg.type === "error") throw new Error(msg.message);
      } catch (err) {
        // If it's a genuine SSE parse issue, ignore; otherwise re-throw.
        if (err instanceof SyntaxError) continue;
        throw err;
      }
    }
  }
}

export interface FlythroughResponse {
  status: string;
  flythrough_url: string;
  request_id: string;
  video_length: number;
  ddim_steps: number;
}

/**
 * POST a still image to the ViewCrafter deployment. Returns a URL to the
 * generated 360° flythrough MP4. Takes ~60 s on a warm L4, ~3–4 min on a
 * cold container (first request pays for weight download).
 */
export async function generateFlythrough(
  frame: Blob,
): Promise<FlythroughResponse> {
  if (!VIEWCRAFTER_URL) {
    throw new Error(
      "ViewCrafter is not configured. Set VITE_VIEWCRAFTER_URL in web/.env.local.",
    );
  }
  const form = new FormData();
  form.append("image", frame, "frame.png");
  const res = await fetch(`${VIEWCRAFTER_URL}/flythrough`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let msg = `flythrough failed: ${res.status}`;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j.detail) msg = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return (await res.json()) as FlythroughResponse;
}

export async function analyze(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  // Backend endpoint expects the multipart part to be named "video".
  form.append("video", file, file.name);
  const res = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let msg = `analyze failed: ${res.status}`;
    try {
      const j = (await res.json()) as { error?: string; detail?: string };
      if (j.detail) msg = j.detail;
      else if (j.error) msg = j.error;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return (await res.json()) as AnalyzeResponse;
}
