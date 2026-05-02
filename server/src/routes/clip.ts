// GET /api/clip/:filename - opaque proxy for OSS-backed videos.
//
// Security model:
//   - The OSS bucket is fully private; only this server (which holds the
//     RAM-account AccessKey) can read it.
//   - The browser only ever sees `/api/clip/<id>.mp4`. The OSS bucket,
//     region, key prefix, and signature are never exposed.
//   - Edge cache (Aliyun ESA) sits in front and serves the vast majority
//     of hits without ever reaching this handler. Path-level cache rule:
//       /api/clip/* → 30 days
//
// Range handling:
//   - We buffer the whole object once (clip-cache), then satisfy any
//     Range request by slicing the in-memory Buffer. This sidesteps the
//     concurrent-Range stream complications of an upstream-pass-through
//     model.
//
// Lookup:
//   - The id encoded in the URL is matched against `gallery[].id` (and,
//     for future-proofing, `hero-primary` for the hero video) in the DB
//     row. Only entries with `kind === "oss"` are servable; others
//     return 404 because they should be loaded directly (local) or from
//     their public CDN (external).

import { Router } from "express";
import rateLimit from "express-rate-limit";
import { ENV } from "../env.js";
import { db } from "../db.js";
import { DEFAULT_CONFIG_PAYLOAD } from "../default-config.js";
import { fetchClip, type OssRef } from "../clip-cache.js";
import type { SiteConfigInput } from "../schema.js";

interface ConfigRow {
  payload: string;
}

/** Linear scan of gallery + hero in the stored config. Cheap (40 items)
 *  and avoids holding any state, so admin edits land instantly. */
function findOssRef(id: string): OssRef | null {
  const row = db
    .prepare(`SELECT payload FROM site_config WHERE id = 1 LIMIT 1`)
    .get() as ConfigRow | undefined;

  let payload: SiteConfigInput;
  if (row) {
    try {
      payload = JSON.parse(row.payload) as SiteConfigInput;
    } catch {
      payload = DEFAULT_CONFIG_PAYLOAD as unknown as SiteConfigInput;
    }
  } else {
    payload = DEFAULT_CONFIG_PAYLOAD as unknown as SiteConfigInput;
  }

  if (id === "hero-primary" && payload.hero.primaryVideo.kind === "oss") {
    const v = payload.hero.primaryVideo;
    return {
      bucket: v.bucket ?? ENV.OSS_BUCKET!,
      key: v.key,
    };
  }
  for (const c of payload.gallery) {
    if (c.id === id && c.src.kind === "oss") {
      return {
        bucket: c.src.bucket ?? ENV.OSS_BUCKET!,
        key: c.src.key,
      };
    }
  }
  return null;
}

// A page load can request up to ~40 clips at once; bump the default
// public-API limit by 10x for this endpoint.
const clipLimiter = rateLimit({
  windowMs: ENV.PUBLIC_API_RATE_WINDOW_MS,
  max: ENV.PUBLIC_API_RATE_LIMIT * 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "too_many_requests" },
});

const FILENAME_RE = /^([a-zA-Z0-9_-]+)\.mp4$/;

export const clipRouter = Router();

clipRouter.get("/api/clip/:filename", clipLimiter, async (req, res) => {
  const filename = req.params.filename ?? "";
  const m = FILENAME_RE.exec(filename);
  if (!m) {
    res.status(400).json({ error: "invalid_filename" });
    return;
  }
  const id = m[1]!;

  const ref = findOssRef(id);
  if (!ref) {
    // Negative lookups are cached short-term by ESA so a typo doesn't
    // hammer the DB on every visit.
    res.setHeader("Cache-Control", "public, max-age=60");
    res.status(404).json({ error: "clip_not_found" });
    return;
  }

  const clip = await fetchClip(ref);
  if (!clip) {
    res.status(502).json({ error: "upstream_unavailable" });
    return;
  }

  // Conditional GET: skip body if the browser already has it.
  const ifNoneMatch = req.headers["if-none-match"];
  if (clip.etag && typeof ifNoneMatch === "string" && ifNoneMatch === clip.etag) {
    res.setHeader("ETag", clip.etag);
    res.status(304).end();
    return;
  }

  res.setHeader("Content-Type", clip.contentType);
  res.setHeader("Accept-Ranges", "bytes");
  if (clip.etag) res.setHeader("ETag", clip.etag);
  if (clip.lastModified) res.setHeader("Last-Modified", clip.lastModified);
  // Browser cache 1d, shared cache (ESA) 30d. ESA console rule can
  // override s-maxage upward if desired.
  res.setHeader(
    "Cache-Control",
    "public, max-age=86400, s-maxage=2592000, immutable",
  );
  // Never let an intermediary turn this into a download.
  res.setHeader("Content-Disposition", "inline");

  const total = clip.bytes.byteLength;
  const range = req.headers.range;
  if (typeof range === "string" && range.startsWith("bytes=")) {
    const rangeMatch = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!rangeMatch) {
      res.setHeader("Content-Range", `bytes */${total}`);
      res.status(416).end();
      return;
    }
    const startStr = rangeMatch[1] ?? "";
    const endStr = rangeMatch[2] ?? "";
    let start: number;
    let end: number;
    if (startStr === "" && endStr !== "") {
      // suffix range: bytes=-N -> last N bytes
      const suffix = parseInt(endStr, 10);
      if (Number.isNaN(suffix) || suffix <= 0) {
        res.setHeader("Content-Range", `bytes */${total}`);
        res.status(416).end();
        return;
      }
      start = Math.max(0, total - suffix);
      end = total - 1;
    } else {
      start = startStr === "" ? 0 : parseInt(startStr, 10);
      end = endStr === "" ? total - 1 : parseInt(endStr, 10);
    }
    if (
      Number.isNaN(start) ||
      Number.isNaN(end) ||
      start < 0 ||
      end >= total ||
      start > end
    ) {
      res.setHeader("Content-Range", `bytes */${total}`);
      res.status(416).end();
      return;
    }
    const chunk = clip.bytes.subarray(start, end + 1);
    res.status(206);
    res.setHeader("Content-Range", `bytes ${start}-${end}/${total}`);
    res.setHeader("Content-Length", String(chunk.byteLength));
    res.end(chunk);
    return;
  }

  res.status(200);
  res.setHeader("Content-Length", String(total));
  res.end(clip.bytes);
});
