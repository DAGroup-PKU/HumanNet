// Seeds (or resets) the admin user and the site_config row using the
// bundled defaults from src/lib/config-defaults.ts. Safe to re-run.
//
// Usage:
//   npm --prefix server run seed
//   ADMIN_USERNAME=alice ADMIN_PASSWORD=hunter2 npm --prefix server run seed

import bcrypt from "bcryptjs";
import { ENV } from "./env.js";
import { db } from "./db.js";
import { DEFAULT_CONFIG_PAYLOAD } from "./default-config.js";

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

function upsertConfig() {
  const payload = JSON.stringify(DEFAULT_CONFIG_PAYLOAD);
  const existing = db.prepare(`SELECT id FROM site_config WHERE id = 1`).get();

  if (existing) {
    console.log(
      "· site_config row already present — leaving payload untouched.",
    );
    return;
  }
  db.prepare(
    `INSERT INTO site_config (id, payload, updated_by) VALUES (1, ?, 'seed')`,
  ).run(payload);
  console.log("· site_config seeded with bundled defaults.");
}

console.log(`Seeding ${ENV.DB_PATH} ...`);
upsertAdmin();
upsertConfig();
console.log("✓ Seed complete.");
