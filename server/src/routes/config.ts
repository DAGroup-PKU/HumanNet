import { Router } from "express";
import { db } from "../db.js";
import { DEFAULT_CONFIG_PAYLOAD } from "../default-config.js";
import { requireAdmin } from "../auth.js";
import { SiteConfigSchema } from "../schema.js";

export const configRouter = Router();

interface ConfigRow {
  payload: string;
  updated_at: string;
}

function readConfig(): { payload: unknown; updated_at: string | null } {
  const row = db
    .prepare(
      `SELECT payload, updated_at FROM site_config WHERE id = 1 LIMIT 1`,
    )
    .get() as ConfigRow | undefined;

  if (!row) {
    return { payload: DEFAULT_CONFIG_PAYLOAD, updated_at: null };
  }
  try {
    return { payload: JSON.parse(row.payload), updated_at: row.updated_at };
  } catch {
    // Corrupt payload — fall back to bundled defaults rather than 500ing.
    return { payload: DEFAULT_CONFIG_PAYLOAD, updated_at: row.updated_at };
  }
}

// Public read.
configRouter.get("/api/config", (_req, res) => {
  const { payload, updated_at } = readConfig();
  res.json({
    ...(payload as object),
    updatedAt: updated_at ?? new Date(0).toISOString(),
  });
});

// Admin write — validates the entire object atomically.
configRouter.put("/api/config", requireAdmin, (req, res) => {
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

  const fresh = readConfig();
  res.json({
    ...(fresh.payload as object),
    updatedAt: fresh.updated_at,
  });
});
