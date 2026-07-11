/**
 * POST /api/analyze
 *
 * Body: { key: string }
 *
 * Kicks off the Modal pipeline for a previously-uploaded R2 object.
 *
 * We pass the R2 presigned GET URL to Modal so the pipeline can pull the
 * video directly — cheaper than streaming bytes through the edge.
 *
 * The response mode depends on what Modal exposes:
 *   - If Modal has /analyze/sse, we upgrade the client to SSE and relay
 *     progress events one-for-one.
 *   - Otherwise we POST to /analyze and pipe the JSON body back.
 *
 * Vercel Node runtime — long-running fetch relay + streaming.
 */
import { Hono } from "hono";
import { handle } from "hono/vercel";
import { stream } from "hono/streaming";
import { z } from "zod";
import { corsMiddleware } from "../src/cors.js";
import { getEnv } from "../src/env.js";
import { presignedGetUrl } from "../src/r2.js";

export const config = {
  runtime: "nodejs",
  // Modal cold starts + full pipeline can push past the default 10s.
  maxDuration: 60,
};

const bodySchema = z.object({
  key: z.string().min(1).max(512),
  /** If true and the client's Accept header includes text/event-stream,
   *  we relay Modal's SSE progress. Defaults to auto-detect. */
  sse: z.boolean().optional(),
});

const app = new Hono().basePath("/api");

app.use("*", corsMiddleware());

app.post("/analyze", async (c) => {
  const raw = await c.req.json().catch(() => ({}));
  const parsed = bodySchema.safeParse(raw);
  if (!parsed.success) {
    return c.json({ error: "invalid body", details: parsed.error.issues }, 400);
  }
  const { key } = parsed.data;
  const env = getEnv();

  const videoUrl = await presignedGetUrl(key, { expiresIn: 3600 });

  const acceptsSse =
    parsed.data.sse ??
    (c.req.header("accept") ?? "").includes("text/event-stream");

  const modalUrl = new URL(
    acceptsSse ? "/analyze/sse" : "/analyze",
    env.MODAL_PIPELINE_URL,
  ).toString();

  const modalRes = await fetch(modalUrl, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: acceptsSse ? "text/event-stream" : "application/json",
    },
    body: JSON.stringify({ video_url: videoUrl, key }),
  });

  if (!modalRes.ok || !modalRes.body) {
    const detail = await modalRes.text().catch(() => "");
    return c.json(
      { error: "modal pipeline error", status: modalRes.status, detail },
      502,
    );
  }

  // JSON path: forward as-is.
  const contentType = modalRes.headers.get("content-type") ?? "";
  if (!acceptsSse || !contentType.includes("text/event-stream")) {
    const text = await modalRes.text();
    c.header("content-type", contentType || "application/json");
    return c.body(text);
  }

  // SSE path: pipe Modal's stream to the client.
  c.header("content-type", "text/event-stream");
  c.header("cache-control", "no-cache, no-transform");
  c.header("connection", "keep-alive");
  c.header("x-accel-buffering", "no");

  return stream(c, async (writable) => {
    const reader = modalRes.body!.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) await writable.write(value);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      await writable.write(
        new TextEncoder().encode(
          `event: error\ndata: ${JSON.stringify({ message: msg })}\n\n`,
        ),
      );
    } finally {
      reader.releaseLock();
    }
  });
});

app.get("/analyze", (c) =>
  c.json({ error: "method not allowed, use POST" }, 405),
);

export default handle(app);
