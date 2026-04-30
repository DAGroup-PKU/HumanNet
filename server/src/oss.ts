// Aliyun OSS signing layer.
//
// Goal: turn a {kind:"oss", bucket?, key} reference (the shape stored in
// the DB) into a short-TTL signed URL the browser can play, without ever
// exposing the storage credentials or even the static object URL.
//
// Lifecycle:
//   - The first signOssRef() call lazily constructs the ali-oss client
//     using OSS_REGION / OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET. If
//     any of those env vars are missing, OSS_ENABLED is false and the
//     function returns a clearly-broken placeholder URL plus a one-time
//     warning - the page still renders, the broken video is obvious.
//   - For multi-bucket setups we cache one client per bucket name. The
//     default bucket comes from OSS_BUCKET; refs may override it.
//
// Hardening notes (intentionally not implemented in code):
//   - The bucket should be set to "private" ACL in the OSS console.
//   - A Referer whitelist on the bucket gives a second layer of defence
//     so even leaked signed URLs only work from the official site.
//   - The RAM credentials in env should be a sub-account with only
//     `oss:GetObject` on the one bucket. NEVER use the root AK.

import OSS from "ali-oss";
import { ENV, OSS_ENABLED } from "./env.js";

let warnedMissing = false;
const clientCache = new Map<string, OSS>();

function getClient(bucket: string): OSS {
  const cached = clientCache.get(bucket);
  if (cached) return cached;

  const client = new OSS({
    region: ENV.OSS_REGION!,
    accessKeyId: ENV.OSS_ACCESS_KEY_ID!,
    accessKeySecret: ENV.OSS_ACCESS_KEY_SECRET!,
    bucket,
    endpoint: ENV.OSS_ENDPOINT,
    secure: ENV.OSS_SECURE,
    // 30-second timeout per operation; signing is local but the SDK
    // sometimes warms up region info on the first call.
    timeout: 30_000,
  });
  clientCache.set(bucket, client);
  return client;
}

export interface OssVideoRef {
  bucket?: string;
  key: string;
}

/** Produce a short-TTL signed URL for the given OSS object. Falls back
 *  to a visibly-broken placeholder when credentials are missing - this
 *  keeps the page from rendering an empty video element while making
 *  the misconfiguration loud in the network tab. */
export async function signOssRef(ref: OssVideoRef): Promise<string> {
  if (!OSS_ENABLED) {
    if (!warnedMissing) {
      // eslint-disable-next-line no-console
      console.warn(
        `[oss] signOssRef called for key="${ref.key}" but OSS is not configured; returning placeholder.`,
      );
      warnedMissing = true;
    }
    return `data:text/plain;charset=utf-8,${encodeURIComponent(
      `OSS not configured (key=${ref.key})`,
    )}`;
  }

  const bucket = ref.bucket ?? ENV.OSS_BUCKET!;
  const client = getClient(bucket);
  try {
    const url = await client.asyncSignatureUrl(ref.key, {
      expires: ENV.OSS_URL_TTL_SECONDS,
      method: "GET",
      // Force inline playback rather than triggering a download - belt
      // & braces with the frontend `controlsList="nodownload"`.
      response: { "content-disposition": "inline" },
    });
    return url;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(
      `[oss] failed to sign url bucket=${bucket} key=${ref.key}:`,
      err,
    );
    return `data:text/plain;charset=utf-8,${encodeURIComponent(
      `OSS sign failed (key=${ref.key})`,
    )}`;
  }
}
