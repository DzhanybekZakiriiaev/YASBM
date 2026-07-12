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

export interface BallisticAudit {
  eligible: boolean;
  reason: string;
  sigma: number;
}

/** One detected + audited object (YOLO class, tracked across frames). */
export interface ObjectReport {
  object_id: number;
  label: string;
  frames_present: number;
  member_track_ids: number[];
  /** frame index (string key) -> [x0, y0, x1, y1] normalized to [0,1] */
  boxes_norm: Record<string, [number, number, number, number]>;
  ballistic: BallisticAudit;
  morph_score: number;
  class_flicker: number;
  rigidity_cv: number;
  box_jerk: number;
  verdict:
    | "consistent"
    | "borderline"
    | "implausible"
    | "agent"
    | "static"
    | "morphing";
}

/**
 * One 3D prop placement for a detected object. Backend resolves a GLB from
 * Poly Pizza when possible (`glb_url` non-null); otherwise the client renders
 * a procedural proxy. `position` is the CENTER of the object's 3D bbox in
 * world metres (camera-at-origin frame), `scale` its approximate dims [w,h,d].
 */
export interface PropPlacement {
  object_id: number;
  label: string;
  position: Vec3;
  scale: Vec3;
  yaw_deg: number;
  glb_url: string | null;
  source: string; // "polypizza" | "none"
  crop_url: string | null;
}

export interface AnalyzeResponse {
  status: string;
  tracks: Track[];
  residuals: Residual[];
  verdict_score: number;
  verdict_basis?: "objects" | "grid_fallback" | null;
  objects?: ObjectReport[];
  max_morph_score?: number;
  point_cloud_url: string | null;
  dynamic_points_url?: string | null;
  frame_width?: number | null;
  frame_height?: number | null;
  fps?: number | null;
  duration_s?: number | null;
  props?: PropPlacement[] | null;
  error?: string | null;
}

/** Error carrying the HTTP status so callers can branch (e.g. 503). */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * The analyze response has no request_id field today — but every artifact
 * URL embeds it as the path segment after `/artifacts/`. Extract it from
 * e.g. `https://host/artifacts/<request_id>/points.ply`.
 */
export function extractRequestId(url: string | null | undefined): string | null {
  if (!url) return null;
  const m = /\/artifacts\/([^/?#]+)/.exec(url);
  return m ? m[1] : null;
}

export interface HeroResponse {
  status: string;
  glb_url: string;
}

/**
 * Ask the backend to generate a hero 3D asset (Tripo image-to-3D from the
 * object's crop). Can take up to ~4 minutes — no timeout, just await.
 * Throws ApiError with status 503 when Tripo is not configured.
 */
export async function generateHero(
  requestId: string,
  objectId: number,
): Promise<HeroResponse> {
  const res = await fetch(`${API_BASE_URL}/hero`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, object_id: objectId }),
  });
  if (!res.ok) {
    let msg = `hero generation failed: ${res.status}`;
    try {
      const j = (await res.json()) as { detail?: string; error?: string };
      if (j.detail) msg = j.detail;
      else if (j.error) msg = j.error;
    } catch {
      /* ignore */
    }
    throw new ApiError(msg, res.status);
  }
  return (await res.json()) as HeroResponse;
}

export async function health(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return (await res.json()) as { status: string };
}

export interface VerdictObjectSummary {
  label: string;
  verdict: string;
  sigma: number;
  morph_score: number;
  reason: string;
}

export interface VerdictRequest {
  verdict_score: number;
  residuals: Residual[];
  clip_duration_s?: number;
  verdict_basis?: string;
  objects?: VerdictObjectSummary[];
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
