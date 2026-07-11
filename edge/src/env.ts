import { z } from "zod";

/**
 * Runtime env validation.
 *
 * Vercel injects env vars from the project settings (see README.md).
 * Local dev reads them from .env via `vercel dev`.
 *
 * We validate lazily (via getEnv()) so importing a module doesn't blow up
 * at build time when Vercel bundles the function.
 */
const envSchema = z.object({
  MODAL_PIPELINE_URL: z.string().url(),
  R2_ACCOUNT_ID: z.string().min(1),
  R2_ACCESS_KEY_ID: z.string().min(1),
  R2_SECRET_ACCESS_KEY: z.string().min(1),
  R2_BUCKET: z.string().min(1),
  ANTHROPIC_API_KEY: z.string().min(1),
  ALLOWED_ORIGIN: z.string().url(),
});

export type Env = z.infer<typeof envSchema>;

let cached: Env | null = null;

export function getEnv(): Env {
  if (cached) return cached;
  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const missing = parsed.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("; ");
    throw new Error(`Invalid environment: ${missing}`);
  }
  cached = parsed.data;
  return cached;
}
