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

function optional(name: string): string | undefined {
  const v = process.env[name];
  return v && v.length > 0 ? v : undefined;
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
  CORS_ORIGIN: process.env.CORS_ORIGIN ?? "http://localhost:5173",

  // ─── Aliyun OSS ───────────────────────────────────────────────────
  // Leave any of these unset to disable OSS signing. Stored videos with
  // kind:"oss" then fall back to a clearly-broken placeholder URL and
  // a warning is logged once at boot. Filling them in switches signing
  // on without code changes.
  OSS_REGION: optional("OSS_REGION"),
  OSS_BUCKET: optional("OSS_BUCKET"),
  OSS_ACCESS_KEY_ID: optional("OSS_ACCESS_KEY_ID"),
  OSS_ACCESS_KEY_SECRET: optional("OSS_ACCESS_KEY_SECRET"),
  // Use the *internal* endpoint when the server runs inside Aliyun (free
  // egress), the public endpoint otherwise. Optional - the SDK derives
  // an endpoint from `region` when omitted.
  OSS_ENDPOINT: optional("OSS_ENDPOINT"),
  // TTL for signed playback URLs. 1h is a reasonable default - long
  // enough that a casual visitor watching the page can replay the video,
  // short enough that a leaked link expires before it spreads.
  OSS_URL_TTL_SECONDS: Number(process.env.OSS_URL_TTL_SECONDS ?? 60 * 60),
  // Use HTTPS for signed URLs. Should always be true in prod.
  OSS_SECURE: (process.env.OSS_SECURE ?? "true").toLowerCase() !== "false",

  // ─── Public API rate limiter ──────────────────────────────────────
  PUBLIC_API_RATE_LIMIT: Number(process.env.PUBLIC_API_RATE_LIMIT ?? 60),
  PUBLIC_API_RATE_WINDOW_MS: Number(
    process.env.PUBLIC_API_RATE_WINDOW_MS ?? 60_000,
  ),
} as const;

export const OSS_ENABLED = Boolean(
  ENV.OSS_REGION &&
    ENV.OSS_BUCKET &&
    ENV.OSS_ACCESS_KEY_ID &&
    ENV.OSS_ACCESS_KEY_SECRET,
);

export const IS_PROD = process.env.NODE_ENV === "production";

if (IS_PROD && ENV.JWT_SECRET.startsWith("dev-only-")) {
  // eslint-disable-next-line no-console
  console.error(
    "[FATAL] JWT_SECRET is still the dev placeholder. Set a strong secret before starting in production.",
  );
  process.exit(1);
}

if (!OSS_ENABLED) {
  // eslint-disable-next-line no-console
  console.warn(
    "[oss] OSS credentials missing — kind:'oss' video refs will not be signed. Set OSS_REGION / OSS_BUCKET / OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET to enable.",
  );
}

void required;
