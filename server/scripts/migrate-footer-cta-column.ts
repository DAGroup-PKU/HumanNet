// One-shot migration: ensure the live `site_config.footer.columns` array
// contains the "Get involved" CTA column (waitlist + arXiv). Idempotent —
// safe to re-run on any database. Used to roll out the Round-15 footer
// change without wiping admin-edited content (which a full reseed would).
//
// Runs out of band of the seed script because seed.ts only overwrites the
// payload when it fails schema validation. Adding a column doesn't break
// the schema, so the seeder leaves the row alone — that's by design,
// admin edits are precious.
//
// Usage:
//   npx tsx server/scripts/migrate-footer-cta-column.ts          # local dev DB
//   DB_PATH=/opt/ego-sites/server/site.db \
//     npx tsx server/scripts/migrate-footer-cta-column.ts        # production
//
// Reports one of:
//   - already present (no-op)
//   - inserted (writes back, prints diff summary)
//   - schema-mismatch (refuses to write — re-run `npm run seed` first)
import { db } from "../src/db.js";
import { ENV } from "../src/env.js";
import { SiteConfigSchema } from "../src/schema.js";

const CTA_COLUMN = {
  id: "get-involved",
  title: "Get involved",
  items: [
    { label: "Join the waitlist", href: "$waitlist", external: true },
    { label: "Read the paper", href: "$arxiv", external: true },
  ],
} as const;

interface Row {
  payload: string;
}

const row = db
  .prepare(`SELECT payload FROM site_config WHERE id = 1`)
  .get() as Row | undefined;

if (!row) {
  console.error(
    `[migrate-footer-cta] no site_config row at ${ENV.DB_PATH} — run \`npm run seed\` first.`,
  );
  process.exit(2);
}

let parsed: unknown;
try {
  parsed = JSON.parse(row.payload);
} catch (err) {
  console.error("[migrate-footer-cta] payload is not valid JSON:", err);
  process.exit(2);
}

const check = SiteConfigSchema.safeParse(parsed);
if (!check.success) {
  console.error(
    "[migrate-footer-cta] payload no longer parses against the current schema. Run `npm --prefix server run seed` to migrate first.",
  );
  for (const issue of check.error.issues.slice(0, 6)) {
    console.error(`  · ${issue.path.join(".")} → ${issue.message}`);
  }
  process.exit(2);
}

const config = check.data;
const existingIds = new Set(config.footer.columns.map((c) => c.id));

if (existingIds.has(CTA_COLUMN.id)) {
  console.log(
    `[migrate-footer-cta] column "${CTA_COLUMN.id}" already present — leaving payload untouched.`,
  );
  process.exit(0);
}

const next = {
  ...config,
  footer: {
    ...config.footer,
    columns: [...config.footer.columns, CTA_COLUMN],
  },
};

const recheck = SiteConfigSchema.safeParse(next);
if (!recheck.success) {
  console.error(
    "[migrate-footer-cta] proposed payload fails schema validation, refusing to write:",
    recheck.error.issues,
  );
  process.exit(2);
}

db.prepare(
  `UPDATE site_config SET payload = ?, updated_at = datetime('now'), updated_by = 'migrate-footer-cta' WHERE id = 1`,
).run(JSON.stringify(next));

console.log(
  `[migrate-footer-cta] inserted "${CTA_COLUMN.id}" column with ${CTA_COLUMN.items.length} items into ${ENV.DB_PATH}.`,
);
console.log(
  `  preview: ${CTA_COLUMN.items.map((i) => `${i.label} → ${i.href}`).join(" · ")}`,
);
