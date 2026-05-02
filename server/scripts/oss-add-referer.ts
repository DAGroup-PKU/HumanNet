// Idempotent OSS bucket Referer-whitelist updater.
//
//   npm --prefix server run oss:add-referer -- <referer1> [referer2] ...
//
// Examples:
//   npm --prefix server run oss:add-referer -- 'https://ego.ai100rank.top/*'
//   npm --prefix server run oss:add-referer -- 'https://*.ai100rank.top/*'
//
// Reads the current bucket Referer whitelist via ali-oss SDK, merges in
// the requested entries (de-dup by exact match), writes back, and prints
// the diff. `allowEmptyReferer` is preserved as-is (we never flip it from
// false→true; that would be a security regression).
//
// Permissions: the AccessKey used here must have
//   oss:GetBucketReferer + oss:PutBucketReferer
// on the target bucket. Standard project AK already has these.
//
// Why this exists: when a new public domain comes online (e.g. moving
// from `localhost` dev → `ego.ai100rank.top` prod), the bucket whitelist
// has to be expanded or all OSS-served videos 403 with
// "You are denied by bucket referer policy" (Lesson L0029, L0038).
// Doing it via console works once; doing it via this script is
// reproducible + auditable + diff-able in commit history.
//
// Console fallback (when AccessKey lacks bucket-policy permissions):
//   1. Aliyun OSS 控制台 → Bucket 列表 → ss-oss-sites
//   2. 数据安全 → 防盗链
//   3. Allow empty Referer  : 关 / OFF
//      Referer 白名单        : add 'https://ego.ai100rank.top/*'
//                              (and 'https://*.ai100rank.top/*' for
//                               future subdomains)
//   4. 保存 / Save
import OSS from "ali-oss";
import { ENV, OSS_ENABLED } from "../src/env.js";

if (!OSS_ENABLED) {
  console.error(
    "[oss-add-referer] env vars missing — need OSS_REGION/BUCKET/AK/AK_SECRET",
  );
  process.exit(2);
}

const newReferers = process.argv.slice(2);
if (newReferers.length === 0) {
  console.error(
    "usage: npm --prefix server run oss:add-referer -- <referer> [...]\n" +
      "  example: npm --prefix server run oss:add-referer -- 'https://ego.ai100rank.top/*'",
  );
  process.exit(2);
}

const client = new OSS({
  region: ENV.OSS_REGION!,
  accessKeyId: ENV.OSS_ACCESS_KEY_ID!,
  accessKeySecret: ENV.OSS_ACCESS_KEY_SECRET!,
  bucket: ENV.OSS_BUCKET!,
  secure: ENV.OSS_SECURE,
});

interface BucketRefererResult {
  allowEmptyReferer: boolean;
  referers: string[];
}

console.log(`[oss-add-referer] bucket = ${ENV.OSS_BUCKET}`);

// ali-oss types declare getBucketReferer() but the actual response shape
// is { allowEmptyReferer: boolean, referers: string[] }; we re-shape it
// defensively in case the SDK differs.
let before:
  | (BucketRefererResult & { res?: unknown })
  | { allowEmpty: boolean; referer: string[] };
try {
  before = (await client.getBucketReferer(
    ENV.OSS_BUCKET!,
  )) as unknown as typeof before;
} catch (err: unknown) {
  const e = err as { status?: number; code?: string; message?: string };
  if (e.status === 403) {
    console.error(
      "\n[oss-add-referer] ❌ AccessKey lacks bucket-management permissions.",
    );
    console.error(
      "  This script needs `oss:GetBucketReferer` + `oss:PutBucketReferer`",
    );
    console.error(
      "  on the bucket. The runtime AccessKey only has `oss:GetObject` for",
    );
    console.error(
      "  least-privilege URL signing (recommended). To run this script:",
    );
    console.error("    a) Use a higher-privilege bucket-admin AccessKey, OR");
    console.error(
      "    b) Update the whitelist via the Aliyun OSS console — see comment",
    );
    console.error(
      "       at the top of this file for the click-through.\n",
    );
    process.exit(2);
  }
  throw err;
}

const allowEmpty =
  "allowEmptyReferer" in before
    ? before.allowEmptyReferer
    : (before as { allowEmpty: boolean }).allowEmpty;
const currentReferers: string[] =
  "referers" in before
    ? before.referers ?? []
    : (before as { referer: string[] }).referer ?? [];

console.log(`  allowEmptyReferer: ${allowEmpty}`);
console.log(`  current referers (${currentReferers.length}):`);
for (const r of currentReferers) console.log(`    - ${r}`);

const existing = new Set(currentReferers);
const toAdd = newReferers.filter((r) => !existing.has(r));
if (toAdd.length === 0) {
  console.log(
    `[oss-add-referer] all ${newReferers.length} requested referers already present — no-op.`,
  );
  process.exit(0);
}

const merged = [...currentReferers, ...toAdd];
console.log();
console.log(`[oss-add-referer] adding ${toAdd.length} new referer(s):`);
for (const r of toAdd) console.log(`    + ${r}`);

await client.putBucketReferer(ENV.OSS_BUCKET!, allowEmpty, merged);

console.log();
console.log("[oss-add-referer] write succeeded. New whitelist:");
for (const r of merged) console.log(`    - ${r}`);
console.log(`  allowEmptyReferer: ${allowEmpty} (unchanged)`);
