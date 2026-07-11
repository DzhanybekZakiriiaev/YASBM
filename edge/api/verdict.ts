/**
 * POST /api/verdict
 *
 * Body:
 *   {
 *     verdict_score: number,          // 0..1, higher = more implausible
 *     residuals: [{ t_s, delta_m, sigma }],
 *     tracks_summary?: string
 *   }
 *
 * Streams the Claude Sonnet 4.5 verdict card as SSE.
 *
 * Vercel Node runtime — Anthropic SDK stream + long-lived response.
 *
 * SSE format: one event per delta.
 *   event: token
 *   data: {"text": "..."}
 *
 *   event: done
 *   data: {"input_tokens": N, "output_tokens": M}
 *
 *   event: error
 *   data: {"message": "..."}
 */
import Anthropic from "@anthropic-ai/sdk";
import { Hono } from "hono";
import { handle } from "hono/vercel";
import { stream } from "hono/streaming";
import { z } from "zod";
import { corsMiddleware } from "../src/cors.js";
import { getEnv } from "../src/env.js";
import { VERDICT_SYSTEM_PROMPT } from "../src/verdict-prompt.js";

export const config = {
  runtime: "nodejs",
  maxDuration: 60,
};

const residualSchema = z.object({
  t_s: z.number(),
  delta_m: z.number(),
  sigma: z.number(),
});

const bodySchema = z.object({
  verdict_score: z.number().min(0).max(1),
  residuals: z.array(residualSchema).min(1).max(2048),
  tracks_summary: z.string().max(2048).optional(),
});

const MODEL = "claude-sonnet-4-5";

const app = new Hono().basePath("/api");

app.use("*", corsMiddleware());

app.post("/verdict", async (c) => {
  const raw = await c.req.json().catch(() => ({}));
  const parsed = bodySchema.safeParse(raw);
  if (!parsed.success) {
    return c.json({ error: "invalid body", details: parsed.error.issues }, 400);
  }
  const { verdict_score, residuals, tracks_summary } = parsed.data;

  const env = getEnv();
  const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });

  // Compact the residuals so we don't burn tokens on a 2000-entry array.
  // We include the peak, the mean, and a coarse sampling.
  const summary = summarizeResiduals(residuals);

  const userMessage = [
    `verdict_score: ${verdict_score.toFixed(3)}`,
    `residual_peak_m: ${summary.peak.toFixed(3)} at t=${summary.peak_t_s.toFixed(2)}s (frame window ${summary.window_start_s.toFixed(2)}-${summary.window_end_s.toFixed(2)}s)`,
    `residual_mean_m: ${summary.mean.toFixed(3)}`,
    `residual_p95_m: ${summary.p95.toFixed(3)}`,
    `sample_points (t_s, delta_m, sigma): ${summary.sample
      .map((r) => `(${r.t_s.toFixed(2)}, ${r.delta_m.toFixed(3)}, ${r.sigma.toFixed(3)})`)
      .join("; ")}`,
    tracks_summary ? `tracks_summary: ${tracks_summary}` : null,
  ]
    .filter(Boolean)
    .join("\n");

  c.header("content-type", "text/event-stream");
  c.header("cache-control", "no-cache, no-transform");
  c.header("connection", "keep-alive");
  c.header("x-accel-buffering", "no");

  return stream(c, async (writable) => {
    const encoder = new TextEncoder();
    const send = (event: string, data: unknown) =>
      writable.write(
        encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
      );

    try {
      const anthropicStream = client.messages.stream({
        model: MODEL,
        max_tokens: 400,
        system: VERDICT_SYSTEM_PROMPT,
        messages: [{ role: "user", content: userMessage }],
      });

      for await (const event of anthropicStream) {
        if (
          event.type === "content_block_delta" &&
          event.delta.type === "text_delta"
        ) {
          await send("token", { text: event.delta.text });
        }
      }

      const final = await anthropicStream.finalMessage();
      await send("done", {
        input_tokens: final.usage.input_tokens,
        output_tokens: final.usage.output_tokens,
        stop_reason: final.stop_reason,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await send("error", { message });
    }
  });
});

app.get("/verdict", (c) =>
  c.json({ error: "method not allowed, use POST" }, 405),
);

interface ResidualSummary {
  peak: number;
  peak_t_s: number;
  window_start_s: number;
  window_end_s: number;
  mean: number;
  p95: number;
  sample: Array<{ t_s: number; delta_m: number; sigma: number }>;
}

function summarizeResiduals(
  residuals: z.infer<typeof residualSchema>[],
): ResidualSummary {
  const first = residuals[0];
  if (!first) {
    // Guarded by zod .min(1), but keep TS happy.
    return {
      peak: 0,
      peak_t_s: 0,
      window_start_s: 0,
      window_end_s: 0,
      mean: 0,
      p95: 0,
      sample: [],
    };
  }

  let peak = first.delta_m;
  let peakIdx = 0;
  let sum = 0;
  for (let i = 0; i < residuals.length; i++) {
    const r = residuals[i]!;
    sum += r.delta_m;
    if (r.delta_m > peak) {
      peak = r.delta_m;
      peakIdx = i;
    }
  }
  const mean = sum / residuals.length;

  const sorted = residuals.map((r) => r.delta_m).sort((a, b) => a - b);
  const p95 = sorted[Math.floor(0.95 * (sorted.length - 1))] ?? peak;

  // Window around the peak (5 samples wide, if available).
  const half = 2;
  const startIdx = Math.max(0, peakIdx - half);
  const endIdx = Math.min(residuals.length - 1, peakIdx + half);
  const windowStart = residuals[startIdx]!.t_s;
  const windowEnd = residuals[endIdx]!.t_s;

  // Downsample to at most 8 evenly spaced points for the model.
  const target = Math.min(8, residuals.length);
  const step = Math.max(1, Math.floor(residuals.length / target));
  const sample: ResidualSummary["sample"] = [];
  for (let i = 0; i < residuals.length && sample.length < target; i += step) {
    sample.push(residuals[i]!);
  }

  return {
    peak,
    peak_t_s: residuals[peakIdx]!.t_s,
    window_start_s: windowStart,
    window_end_s: windowEnd,
    mean,
    p95,
    sample,
  };
}

export default handle(app);
