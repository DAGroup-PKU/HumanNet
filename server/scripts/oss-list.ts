// Diagnostic: list keys in the OSS bucket. NOT used at runtime.
// Requires temporary oss:ListObjects on the RAM policy. Remove that
// permission after diagnosis to keep the runtime AK least-privilege.
import OSS from "ali-oss";
import { ENV, OSS_ENABLED } from "../src/env.js";

if (!OSS_ENABLED) {
  console.error("[oss] env vars missing");
  process.exit(2);
}

const client = new OSS({
  region: ENV.OSS_REGION!,
  accessKeyId: ENV.OSS_ACCESS_KEY_ID!,
  accessKeySecret: ENV.OSS_ACCESS_KEY_SECRET!,
  bucket: ENV.OSS_BUCKET!,
  secure: ENV.OSS_SECURE,
});

const prefix = process.argv[2] ?? "";
const max = Number(process.argv[3] ?? 50);

console.log(`bucket : ${ENV.OSS_BUCKET}`);
console.log(`prefix : "${prefix}"`);
console.log();

const res = await client.list(
  { prefix, "max-keys": max, delimiter: "" },
  {},
);
if (!res.objects || res.objects.length === 0) {
  console.log("(no objects matched)");
  process.exit(0);
}
for (const o of res.objects) {
  const sizeKB = (o.size / 1024).toFixed(1);
  console.log(`${sizeKB.padStart(10)} KB  ${o.name}`);
}
console.log(`\n${res.objects.length} keys shown`);
