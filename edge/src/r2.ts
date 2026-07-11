import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { getEnv } from "./env.js";

/**
 * Cloudflare R2 speaks the S3 API. The endpoint is:
 *   https://<ACCOUNT_ID>.r2.cloudflarestorage.com
 *
 * https://developers.cloudflare.com/r2/api/s3/api/
 * https://developers.cloudflare.com/r2/api/s3/presigned-urls/
 *
 * Region MUST be "auto" for R2.
 */

let cachedClient: S3Client | null = null;

function client(): S3Client {
  if (cachedClient) return cachedClient;
  const env = getEnv();
  cachedClient = new S3Client({
    region: "auto",
    endpoint: `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: env.R2_ACCESS_KEY_ID,
      secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    },
  });
  return cachedClient;
}

const DEFAULT_EXPIRES_SECONDS = 900; // 15 min

/**
 * Presigned PUT URL — browser uploads video bytes directly to R2.
 * Content-Type is enforced client-side; we leave it flexible here.
 */
export async function presignedPutUrl(
  key: string,
  opts: { contentType?: string; expiresIn?: number } = {},
): Promise<string> {
  const env = getEnv();
  const command = new PutObjectCommand({
    Bucket: env.R2_BUCKET,
    Key: key,
    ContentType: opts.contentType,
  });
  return getSignedUrl(client(), command, {
    expiresIn: opts.expiresIn ?? DEFAULT_EXPIRES_SECONDS,
  });
}

/**
 * Presigned GET URL — the Modal pipeline (or the browser) fetches the
 * uploaded video without needing R2 credentials.
 */
export async function presignedGetUrl(
  key: string,
  opts: { expiresIn?: number } = {},
): Promise<string> {
  const env = getEnv();
  const command = new GetObjectCommand({
    Bucket: env.R2_BUCKET,
    Key: key,
  });
  return getSignedUrl(client(), command, {
    expiresIn: opts.expiresIn ?? DEFAULT_EXPIRES_SECONDS,
  });
}

/**
 * Generate a deterministic-but-unique R2 key for an upload.
 * Layout: uploads/<yyyy>/<mm>/<random>.<ext>
 */
export function newUploadKey(ext = "mp4"): string {
  const now = new Date();
  const yyyy = now.getUTCFullYear();
  const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
  // 12 bytes of randomness, base32-ish. Node 20+ has crypto.randomUUID.
  const id = crypto.randomUUID().replace(/-/g, "");
  const cleanExt = ext.replace(/^\.+/, "").toLowerCase().slice(0, 8) || "mp4";
  return `uploads/${yyyy}/${mm}/${id}.${cleanExt}`;
}
