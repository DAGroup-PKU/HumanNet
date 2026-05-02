import { Router } from "express";
import rateLimit from "express-rate-limit";
import { ENV } from "../env.js";
import { db } from "../db.js";
import { DEFAULT_CONFIG_PAYLOAD } from "../default-config.js";
import { requireAdmin } from "../auth.js";
import { SiteConfigSchema, type SiteConfigInput } from "../schema.js";
import { renderPublicConfig } from "../render.js";
import { invalidateAllClips } from "../clip-cache.js";

export const configRouter = Router();

interface ConfigRow {
  payload: string;
  updated_at: string;
}

/** Read the stored config from the DB, or fall back to bundled defaults
 *  if the row is missing / corrupt. We never throw from here — the
 *  public endpoint must be cheap and reliable. */
function readStored(): { payload: SiteConfigInput; updated_at: string | null } {
  const row = db
    .prepare(
      `SELECT payload, updated_at FROM site_config WHERE id = 1 LIMIT 1`,
    )
    .get() as ConfigRow | undefined;

  if (!row) {
    return {
      payload: DEFAULT_CONFIG_PAYLOAD as unknown as SiteConfigInput,
      updated_at: null,
    };
  }
  try {
    return {
      payload: JSON.parse(row.payload) as SiteConfigInput,
      updated_at: row.updated_at,
    };
  } catch {
    return {
      payload: DEFAULT_CONFIG_PAYLOAD as unknown as SiteConfigInput,
      updated_at: row.updated_at,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────
// Public read — rendered shape (signed OSS URLs, resolved footer hrefs).
// ─────────────────────────────────────────────────────────────────────

const publicLimiter = rateLimit({
  windowMs: ENV.PUBLIC_API_RATE_WINDOW_MS,
  max: ENV.PUBLIC_API_RATE_LIMIT,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "too_many_requests" },
});

configRouter.get("/api/config", publicLimiter, (_req, res, next) => {
  try {
    const { payload, updated_at } = readStored();
    const rendered = renderPublicConfig(
      payload,
      updated_at ?? new Date(0).toISOString(),
    );
    // The config payload itself is small and changes only when the
    // admin saves. A short shared-cache TTL plus must-revalidate keeps
    // the cache hot without ever pinning a stale gallery shape.
    res.setHeader(
      "Cache-Control",
      `public, max-age=60, s-maxage=60, must-revalidate`,
    );
    res.json(rendered);
  } catch (err) {
    next(err);
  }
});

// ─────────────────────────────────────────────────────────────────────
// Admin endpoints — raw stored shape (videos as discriminated unions,
// footer hrefs unresolved). The admin SPA needs the raw shape so it can
// edit kind/bucket/key for OSS refs.
// ─────────────────────────────────────────────────────────────────────

configRouter.get("/api/admin/config", requireAdmin, (_req, res) => {
  const { payload, updated_at } = readStored();
  res.json({
    ...payload,
    updatedAt: updated_at ?? new Date(0).toISOString(),
  });
});

configRouter.put("/api/admin/config", requireAdmin, (req, res) => {
  const parsed = SiteConfigSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({
      error: "validation_failed",
      issues: parsed.error.issues.map((i) => ({
        path: i.path.join("."),
        message: i.message,
      })),
    });
    return;
  }

  const json = JSON.stringify(parsed.data);
  const username = req.admin?.username ?? "unknown";

  db.prepare(
    `INSERT INTO site_config (id, payload, updated_at, updated_by)
     VALUES (1, ?, datetime('now'), ?)
     ON CONFLICT(id) DO UPDATE SET
       payload = excluded.payload,
       updated_at = excluded.updated_at,
       updated_by = excluded.updated_by`,
  ).run(json, username);

  // The clip-id → bucket/key mapping just changed. Drop the in-memory
  // bytes cache so a re-mapped id doesn't keep serving stale content.
  // ESA edge cache is not invalidated automatically — admin must purge
  // /api/clip/* via the ESA console after a renaming admin save.
  invalidateAllClips();

  const fresh = readStored();
  res.json({
    ...fresh.payload,
    updatedAt: fresh.updated_at,
  });
});
