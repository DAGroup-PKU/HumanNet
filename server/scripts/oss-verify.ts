// One-shot end-to-end verification for the OSS signing + Referer pipeline.
//
//   npm --prefix server run oss:verify -- <key> [referer]
//
// The script:
//   1. Loads .env from the project root (same as the main server).
//   2. Signs <key> with the live ali-oss client.
//   3. Hits the signed URL three times to prove the layered defences:
//        - no Referer header           -> expect 403 (RefererDenied)
//        - whitelisted Referer header  -> expect 200 (or 404 if key absent)
//        - bogus Referer header        -> expect 403 (RefererDenied)
//   4. Prints a compact verdict line so a human can scan it in 1 second.
//
// It never echoes the AccessKeySecret, only the public URL components.

// Importing env first triggers the custom .env loader in src/env.ts.
import { ENV, OSS_ENABLED } from "../src/env.js";
import { signOssRef } from "../src/oss.js";

const RED = "\x1b[31m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function color(c: string, s: string): string {
  return `${c}${s}${RESET}`;
}

interface ProbeResult {
  status: number;
  size: string;
  serverError: string | null;
}

async function probe(
  url: string,
  referer: string | undefined,
): Promise<ProbeResult> {
  const headers: Record<string, string> = {
    // Pretend to be a normal browser so we exercise the same code path
    // a real <video> tag would.
    "User-Agent":
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    // 1KB byte range = same as the browser's first range request, but
    // doesn't download the whole video. We use GET (not HEAD) because
    // ali-oss signs URLs for GET specifically — HEAD with a GET-signed
    // URL fails signature verification.
    Range: "bytes=0-1023",
  };
  if (referer) headers["Referer"] = referer;

  const res = await fetch(url, { method: "GET", headers });
  // Drain body so we don't leak the connection.
  const body = res.body ? await res.text() : "";
  let serverError: string | null = null;
  if (!res.ok) {
    const code = body.match(/<Code>([^<]+)<\/Code>/)?.[1];
    const msg = body.match(/<Message>([^<]+)<\/Message>/)?.[1];
    serverError = code ? `${code}${msg ? `: ${msg}` : ""}` : null;
  }
  return {
    status: res.status,
    size: res.headers.get("content-length") ?? "-",
    serverError,
  };
}

function expectation(
  label: string,
  actual: number,
  ok: (s: number) => boolean,
  extraHint = "",
): string {
  const marker = ok(actual)
    ? color(GREEN, "✓")
    : color(RED, "✗");
  return `${marker} ${label.padEnd(28)} → HTTP ${actual}${
    extraHint ? `  ${color(DIM, `(${extraHint})`)}` : ""
  }`;
}

async function main() {
  const key = process.argv[2];
  const whitelistRef = process.argv[3] ?? "http://localhost:5173/";

  if (!key) {
    console.error(
      color(
        RED,
        "usage: npm --prefix server run oss:verify -- <key> [whitelistedReferer]",
      ),
    );
    console.error(
      color(
        DIM,
        '  example: npm --prefix server run oss:verify -- "nebula/exo/clip-1.mp4"',
      ),
    );
    process.exit(2);
  }

  if (!OSS_ENABLED) {
    console.error(
      color(
        RED,
        "[oss] env vars missing — set OSS_REGION / OSS_BUCKET / OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET in .env",
      ),
    );
    process.exit(2);
  }

  console.log(color(DIM, "── OSS configuration ────────────────────────────"));
  console.log(`  region : ${ENV.OSS_REGION}`);
  console.log(`  bucket : ${ENV.OSS_BUCKET}`);
  console.log(`  TTL    : ${ENV.OSS_URL_TTL_SECONDS}s`);
  console.log(`  secure : ${ENV.OSS_SECURE}`);
  console.log();

  console.log(color(DIM, "── Signing ──────────────────────────────────────"));
  const url = await signOssRef({ key });
  console.log(`  key            : ${key}`);
  // Print just origin + path; query string is the signature.
  try {
    const u = new URL(url);
    console.log(`  signed url     : ${u.origin}${u.pathname}`);
    console.log(`  query (truncd) : ${u.search.slice(0, 80)}…`);
  } catch {
    console.log(`  signed url     : ${url.slice(0, 80)}…`);
  }
  console.log();

  console.log(color(DIM, "── Probing ──────────────────────────────────────"));
  const noRef = await probe(url, undefined);
  const ok = await probe(url, whitelistRef);
  const evil = await probe(url, "https://evil.example.com/");

  // 200/206 = success (206 is what we expect for our 1KB Range probe),
  // 404 = signing OK but key not in bucket (still counts as auth-pass),
  // 403 = blocked.
  console.log(
    expectation("no Referer (script/wget)", noRef.status, (s) => s === 403, noRef.serverError ?? "should be RefererDenied"),
  );
  console.log(
    expectation(
      `whitelist Referer (${whitelistRef})`,
      ok.status,
      (s) => s === 200 || s === 206 || s === 404,
      ok.status === 206
        ? `${ok.size}B partial content - playback works`
        : ok.status === 404
          ? "key not in bucket, but signature & Referer accepted"
          : ok.serverError ?? "should be 200/206",
    ),
  );
  console.log(
    expectation("evil Referer", evil.status, (s) => s === 403, evil.serverError ?? "should be RefererDenied"),
  );

  console.log();
  const allOk =
    noRef.status === 403 &&
    (ok.status === 200 || ok.status === 206 || ok.status === 404) &&
    evil.status === 403;
  if (allOk) {
    console.log(color(GREEN, "verdict: signing + Referer whitelist working ✓"));
    if (ok.status === 404) {
      console.log(
        color(
          YELLOW,
          "         (key absent in bucket — upload it or use an existing key for a 200 OK)",
        ),
      );
    }
    process.exit(0);
  } else {
    console.log(
      color(
        RED,
        "verdict: pipeline NOT working as expected — see HTTP codes above",
      ),
    );
    if (noRef.status !== 403) {
      console.log(
        color(
          YELLOW,
          "  · noRef returned non-403 — likely 'allow empty Referer' is on, or RAM permissions are wider than oss:GetObject",
        ),
      );
    }
    if (ok.status === 403) {
      console.log(
        color(
          YELLOW,
          `  · whitelist Referer "${whitelistRef}" was rejected — recheck the bucket Referer list (must end with /*)`,
        ),
      );
    }
    if (evil.status !== 403) {
      console.log(
        color(
          YELLOW,
          "  · evil Referer was accepted — Referer whitelist is missing or empty-Referer is allowed",
        ),
      );
    }
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(color(RED, "[oss-verify] crashed:"), err);
  process.exit(1);
});
