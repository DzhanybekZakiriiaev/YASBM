import { cors } from "hono/cors";
import type { MiddlewareHandler } from "hono";
import { getEnv } from "./env.js";

/**
 * CORS middleware locked to the configured web origin.
 *
 * We resolve the origin lazily so a missing env var in build time
 * doesn't crash the bundle — the request handler surfaces the error.
 */
export function corsMiddleware(): MiddlewareHandler {
  return cors({
    origin: (origin) => {
      const allowed = getEnv().ALLOWED_ORIGIN;
      // Reflect the exact origin when it matches; otherwise deny.
      return origin === allowed ? origin : allowed;
    },
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    exposeHeaders: ["Content-Type"],
    maxAge: 86400,
  });
}
