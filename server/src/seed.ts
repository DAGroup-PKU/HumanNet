// Seeds (or resets) the admin user and the site_config row using the
// bundled defaults from server/src/default-config.ts. Safe to re-run.
//
// Usage:
//   npm --prefix server run seed
//   ADMIN_USERNAME=alice ADMIN_PASSWORD=hunter2 npm --prefix server run seed
//
// Migration policy: when the existing site_config row no longer parses
// against the current SiteConfigSchema (e.g. after we evolved the schema
// to add VideoRef and FooterConfig), the seeder re-seeds with the
// bundled defaults rather than leaving a half-broken row in place. The
// previous payload is logged so it can be diff'd if anything important
// was lost.

import bcrypt from "bcryptjs";
import { ENV } from "./env.js";
import { db } from "./db.js";
import { DEFAULT_CONFIG_PAYLOAD } from "./default-config.js";
import { SiteConfigSchema } from "./schema.js";

function upsertAdmin() {
  const hash = bcrypt.hashSync(ENV.ADMIN_PASSWORD, 12);
  const existing = db
    .prepare(`SELECT id FROM users WHERE username = ?`)
    .get(ENV.ADMIN_USERNAME) as { id: number } | undefined;

  if (existing) {
    db.prepare(`UPDATE users SET password_hash = ? WHERE id = ?`).run(
      hash,
      existing.id,
    );
    console.log(
      `· admin user "${ENV.ADMIN_USERNAME}" exists — password reset.`,
    );
  } else {
    db.prepare(
      `INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')`,
    ).run(ENV.ADMIN_USERNAME, hash);
    console.log(`· admin user "${ENV.ADMIN_USERNAME}" created.`);
  }
}

interface ExistingRow {
  payload: string;
  updated_at: string;
}

function upsertConfig() {
  const payloadJson = JSON.stringify(DEFAULT_CONFIG_PAYLOAD);
  const existing = db
    .prepare(`SELECT payload, updated_at FROM site_config WHERE id = 1`)
    .get() as ExistingRow | undefined;

  if (!existing) {
    db.prepare(
      `INSERT INTO site_config (id, payload, updated_by) VALUES (1, ?, 'seed')`,
    ).run(payloadJson);
    console.log("· site_config seeded with bundled defaults.");
    return;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(existing.payload);
  } catch (err) {
    console.warn(
      "· site_config payload is not valid JSON — overwriting with defaults.",
      err,
    );
    db.prepare(
      `UPDATE site_config SET payload = ?, updated_at = datetime('now'), updated_by = 'seed-migrate' WHERE id = 1`,
    ).run(payloadJson);
    return;
  }

  const check = SiteConfigSchema.safeParse(parsed);
  if (check.success) {
    console.log(
      "· site_config row already present and schema-valid — leaving payload untouched.",
    );
    return;
  }

  console.warn(
    "· site_config payload no longer parses against the current schema — running migration.",
  );
  console.warn("  validation issues:");
  for (const issue of check.error.issues.slice(0, 6)) {
    console.warn(`    · ${issue.path.join(".")} → ${issue.message}`);
  }
  if (check.error.issues.length > 6) {
    console.warn(`    · …(+${check.error.issues.length - 6} more)`);
  }
  console.warn(
    `  previous payload (truncated): ${existing.payload.slice(0, 200)}…`,
  );
  db.prepare(
    `UPDATE site_config SET payload = ?, updated_at = datetime('now'), updated_by = 'seed-migrate' WHERE id = 1`,
  ).run(payloadJson);
  console.log("· site_config migrated to new schema.");
}

console.log(`Seeding ${ENV.DB_PATH} ...`);
upsertAdmin();
upsertConfig();
console.log("✓ Seed complete.");
