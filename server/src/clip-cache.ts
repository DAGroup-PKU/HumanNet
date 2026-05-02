// In-memory LRU cache for OSS-backed clips.
//
// Goal: serve every gallery clip from RAM after the first miss. The 40
// curated clips total ~72 MB so the entire dataset fits comfortably; the
// cap exists only as a guard rail for future growth.
//
// Why buffer the whole object instead of streaming?
//   - Clips are tiny (avg ~2 MB). The whole-buffer approach makes Range
//     handling a one-line slice and removes the corner-cases around
//     half-finished streams + concurrent Range requests for the same key.
//   - The fetch path concurrent guard (`inflight`) collapses simultaneous
//     misses for the same key into a single OSS round trip.
//
// Eviction: pure LRU. Map iteration order in V8 is insertion order, so
// `cache.delete(k); cache.set(k, v)` re-promotes a key to "most recently
// used". The oldest insertion is evicted when capacity is reached.

import OSS from "ali-oss";
import { ENV, OSS_ENABLED } from "./env.js";

export interface OssRef {
  bucket: string;
  key: string;
}

export interface CachedClip {
  bytes: Buffer;
  contentType: string;
  /** Quoted -> unquoted ETag from OSS (no surrounding "). */
  etag?: string;
  lastModified?: string;
  fetchedAt: number;
}

const MAX_CACHE_BYTES = Number(
  process.env.CLIP_CACHE_MAX_BYTES ?? 250 * 1024 * 1024, // 250 MB
);
const MAX_PER_OBJECT_BYTES = Number(
  process.env.CLIP_CACHE_MAX_OBJECT_BYTES ?? 50 * 1024 * 1024, // 50 MB
);

const cache = new Map<string, CachedClip>();
let cacheBytes = 0;

const inflight = new Map<string, Promise<CachedClip | null>>();

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
    timeout: 60_000,
  });
  clientCache.set(bucket, client);
  return client;
}

function evictTo(neededBytes: number): void {
  while (
    cacheBytes + neededBytes > MAX_CACHE_BYTES &&
    cache.size > 0
  ) {
    const oldest = cache.keys().next().value as string | undefined;
    if (!oldest) break;
    const entry = cache.get(oldest);
    if (entry) cacheBytes -= entry.bytes.byteLength;
    cache.delete(oldest);
  }
}

function stripETag(raw: string | undefined): string | undefined {
  if (!raw) return undefined;
  return raw.replace(/^"|"$/g, "");
}

function pickHeader(
  headers: unknown,
  name: string,
): string | undefined {
  if (!headers || typeof headers !== "object") return undefined;
  const h = headers as Record<string, string | string[] | undefined>;
  const v = h[name] ?? h[name.toLowerCase()];
  if (Array.isArray(v)) return v[0];
  return v;
}

/** Fetch a clip from OSS (or cache). Returns null when the object is
 *  unreachable (missing creds, network error, bucket denies us). */
export async function fetchClip(ref: OssRef): Promise<CachedClip | null> {
  if (!OSS_ENABLED) return null;
  const cacheKey = `${ref.bucket}/${ref.key}`;

  const hit = cache.get(cacheKey);
  if (hit) {
    cache.delete(cacheKey);
    cache.set(cacheKey, hit);
    return hit;
  }

  const pending = inflight.get(cacheKey);
  if (pending) return pending;

  const promise = (async () => {
    try {
      const client = getClient(ref.bucket);
      const result = await client.get(ref.key);
      const buf = result.content as Buffer;
      const headers = result.res?.headers;
      const entry: CachedClip = {
        bytes: buf,
        contentType:
          pickHeader(headers, "content-type") ?? "application/octet-stream",
        etag: stripETag(pickHeader(headers, "etag")),
        lastModified: pickHeader(headers, "last-modified"),
        fetchedAt: Date.now(),
      };
      // Only cache "small enough" objects; very large files would
      // pin too much RAM if 40 of them landed in cache.
      if (buf.byteLength <= MAX_PER_OBJECT_BYTES) {
        evictTo(buf.byteLength);
        cache.set(cacheKey, entry);
        cacheBytes += buf.byteLength;
      }
      return entry;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(
        `[clip-cache] fetch failed bucket=${ref.bucket} key=${ref.key}:`,
        err,
      );
      return null;
    } finally {
      inflight.delete(cacheKey);
    }
  })();

  inflight.set(cacheKey, promise);
  return promise;
}

export function clipCacheStats() {
  return {
    entries: cache.size,
    bytes: cacheBytes,
    maxBytes: MAX_CACHE_BYTES,
  };
}

/** Clear the entire cache. Exposed for the admin write path so a
 *  freshly-edited gallery doesn't keep serving stale bytes after a
 *  bucket/key swap. */
export function invalidateAllClips(): void {
  cache.clear();
  cacheBytes = 0;
}
