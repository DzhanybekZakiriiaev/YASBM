/**
 * POST /api/upload
 *
 * Returns a presigned R2 PUT URL. The browser uploads the video bytes
 * directly to R2, then calls /api/analyze with the returned key.
 *
 * Body (optional):
 *   { ext?: string, contentType?: string }
 *
 * Response:
 *   { url: string, key: string }
 *
 * Vercel Node runtime — presigning is fast and stateless but pulls in
 * the AWS SDK which is too heavy for the Edge runtime.
 */
import { Hono } from "hono";
import { handle } from "hono/vercel";
import { z } from "zod";
import { corsMiddleware } from "../src/cors.js";
import { newUploadKey, presignedPutUrl } from "../src/r2.js";

export const config = {
  runtime: "nodejs",
};

const bodySchema = z
  .object({
    ext: z.string().max(8).optional(),
    contentType: z.string().max(128).optional(),
  })
  .default({});

const app = new Hono().basePath("/api");

app.use("*", corsMiddleware());

app.post("/upload", async (c) => {
  const raw = await c.req.json().catch(() => ({}));
  const parsed = bodySchema.safeParse(raw);
  if (!parsed.success) {
    return c.json({ error: "invalid body", details: parsed.error.issues }, 400);
  }
  const { ext, contentType } = parsed.data;

  const key = newUploadKey(ext ?? "mp4");
  const url = await presignedPutUrl(key, {
    contentType: contentType ?? "video/mp4",
  });

  return c.json({ url, key });
});

app.get("/upload", (c) =>
  c.json({ error: "method not allowed, use POST" }, 405),
);

export default handle(app);
