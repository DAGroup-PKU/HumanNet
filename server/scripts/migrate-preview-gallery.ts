// One-shot migration: replace the live site_config.gallery with the
// manifest-driven Preview gallery from server/src/preview-gallery.ts.
// Idempotent and safe to re-run. It preserves hero, links, footer, and
// other admin-edited fields.
//
// Usage:
//   npm --prefix server run migrate:preview-gallery
//   DB_PATH=/opt/ego-sites/server/site.db \
//     npm --prefix server run migrate:preview-gallery

import { db } from "../src/db.js";
import { ENV } from "../src/env.js";
import { PREVIEW_GALLERY } from "../src/preview-gallery.js";
import { SiteConfigSchema } from "../src/schema.js";

interface Row {
  payload: string;
}

const row = db
  .prepare(`SELECT payload FROM site_config WHERE id = 1`)
  .get() as Row | undefined;

if (!row) {
  console.error(
    `[migrate-preview-gallery] no site_config row at ${ENV.DB_PATH} — run \`npm --prefix server run seed\` first.`,
  );
  process.exit(2);
}

let parsed: unknown;
try {
  parsed = JSON.parse(row.payload);
} catch (err) {
  console.error("[migrate-preview-gallery] payload is not valid JSON:", err);
  process.exit(2);
}

const check = SiteConfigSchema.safeParse(parsed);
if (!check.success) {
  console.error(
    "[migrate-preview-gallery] payload no longer parses against the current schema. Run `npm --prefix server run seed` to migrate first.",
  );
  for (const issue of check.error.issues.slice(0, 6)) {
    console.error(`  · ${issue.path.join(".")} -> ${issue.message}`);
  }
  process.exit(2);
}

const config = check.data;
const currentKeys = config.gallery.map((clip) =>
  clip.src.kind === "oss" ? clip.src.key : `${clip.src.kind}:${clip.id}`,
);
const nextKeys = PREVIEW_GALLERY.map((clip) => clip.src.key);

if (
  currentKeys.length === nextKeys.length &&
  currentKeys.every((key, index) => key === nextKeys[index])
) {
  console.log(
    `[migrate-preview-gallery] gallery already matches preview manifest (${nextKeys.length} clips) — leaving payload untouched.`,
  );
  process.exit(0);
}

const next = {
  ...config,
  gallery: PREVIEW_GALLERY,
};

const recheck = SiteConfigSchema.safeParse(next);
if (!recheck.success) {
  console.error(
    "[migrate-preview-gallery] proposed payload fails schema validation, refusing to write:",
    recheck.error.issues,
  );
  process.exit(2);
}

db.prepare(
  `UPDATE site_config SET payload = ?, updated_at = datetime('now'), updated_by = 'migrate-preview-gallery' WHERE id = 1`,
).run(JSON.stringify(next));

console.log(
  `[migrate-preview-gallery] replaced gallery in ${ENV.DB_PATH}: ${currentKeys.length} -> ${nextKeys.length} clips.`,
);
console.log("  prefixes: preview/tpv, preview/fpv");
