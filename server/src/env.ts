import { existsSync } from "node:fs";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Tiny .env loader — keeps the dependency surface small (no dotenv).
// Loads `<repo>/.env` if it exists; missing file is fine for prod where
// the host injects vars directly.
const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const dotenvPath = path.join(repoRoot, ".env");

if (existsSync(dotenvPath)) {
  const raw = readFileSync(dotenvPath, "utf8");
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = value;
  }
}

function required(name: string, fallback?: string): string {
  const v = process.env[name] ?? fallback;
  if (!v || v.length === 0) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return v;
}

export const ENV = {
  PORT: Number(process.env.SERVER_PORT ?? 5175),
  DB_PATH: process.env.DB_PATH ?? path.join(repoRoot, "server/data/nebula.db"),
  // JWT_SECRET MUST be overridden in production via env. The fallback only
  // works for local dev so the seed flow doesn't crash.
  JWT_SECRET:
    process.env.JWT_SECRET ??
    "dev-only-do-not-use-in-prod-please-change-me-immediately",
  JWT_TTL_SECONDS: Number(process.env.JWT_TTL_SECONDS ?? 60 * 60 * 8),
  ADMIN_USERNAME: process.env.ADMIN_USERNAME ?? "admin",
  // ADMIN_PASSWORD is consumed by `npm run seed` to (re)set the admin
  // password. Not used at runtime — only the bcrypt hash in the DB is.
  ADMIN_PASSWORD: process.env.ADMIN_PASSWORD ?? "nebula-admin",
  CORS_ORIGIN: process.env.CORS_ORIGIN ?? "http://localhost:5174",
} as const;

export const IS_PROD = process.env.NODE_ENV === "production";

if (IS_PROD && ENV.JWT_SECRET.startsWith("dev-only-")) {
  // eslint-disable-next-line no-console
  console.error(
    "[FATAL] JWT_SECRET is still the dev placeholder. Set a strong secret before starting in production.",
  );
  process.exit(1);
}

// Suppress unused warning if not directly referenced.
void required;
