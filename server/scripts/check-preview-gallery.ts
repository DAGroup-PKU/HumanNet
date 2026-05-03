import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { basename, join } from "node:path";
import { PREVIEW_GALLERY_MANIFEST, PREVIEW_GALLERY } from "../src/preview-gallery.js";

const EXPECTED_TOTAL = 40;
const EXPECTED_BY_DIR = 20;
const ROOT = process.cwd().endsWith("/server") ? ".." : ".";

function fail(message: string): never {
  console.error(`[gallery] ${message}`);
  process.exit(1);
}

function listMp4(dir: string): string[] {
  const full = join(ROOT, dir);
  if (!existsSync(full)) fail(`missing local directory: ${dir}`);
  return readdirSync(full)
    .filter((name) => name.endsWith(".mp4"))
    .sort();
}

function ffprobeDimensions(path: string): { width: number; height: number } {
  const output = execFileSync(
    "ffprobe",
    [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height",
      "-of",
      "csv=s=x:p=0",
      path,
    ],
    { encoding: "utf8" },
  ).trim();
  const [width, height] = output.split("x").map(Number);
  if (!width || !height) fail(`could not read video dimensions: ${path}`);
  return { width, height };
}

if (PREVIEW_GALLERY_MANIFEST.length !== EXPECTED_TOTAL) {
  fail(`manifest has ${PREVIEW_GALLERY_MANIFEST.length} clips; expected ${EXPECTED_TOTAL}`);
}

if (PREVIEW_GALLERY.length !== PREVIEW_GALLERY_MANIFEST.length) {
  fail("rendered PREVIEW_GALLERY length does not match manifest length");
}

const ids = new Set<string>();
const ossKeys = new Set<string>();
for (const clip of PREVIEW_GALLERY_MANIFEST) {
  if (ids.has(clip.id)) fail(`duplicate id: ${clip.id}`);
  if (ossKeys.has(clip.ossKey)) fail(`duplicate OSS key: ${clip.ossKey}`);
  ids.add(clip.id);
  ossKeys.add(clip.ossKey);

  const expectedPrefix = clip.perspective === "exo" ? "preview/tpv/" : "preview/fpv/";
  if (!clip.ossKey.startsWith(expectedPrefix)) {
    fail(`${clip.id} has wrong OSS prefix: ${clip.ossKey}`);
  }
  if (basename(clip.localPath) !== basename(clip.ossKey)) {
    fail(`${clip.id} local filename does not match OSS key filename`);
  }
  if (!existsSync(join(ROOT, clip.localPath))) {
    fail(`${clip.id} missing local source: ${clip.localPath}`);
  }

  const { width, height } = ffprobeDimensions(join(ROOT, clip.localPath));
  if (width !== clip.width || height !== clip.height) {
    fail(`${clip.id} dimension mismatch: manifest ${clip.width}x${clip.height}, file ${width}x${height}`);
  }
}

const tpvFiles = listMp4("media/curation/video/tpv");
const fpvFiles = listMp4("media/curation/video/fpv");
if (tpvFiles.length !== EXPECTED_BY_DIR) {
  fail(`media/curation/video/tpv has ${tpvFiles.length} mp4 files; expected ${EXPECTED_BY_DIR}`);
}
if (fpvFiles.length !== EXPECTED_BY_DIR) {
  fail(`media/curation/video/fpv has ${fpvFiles.length} mp4 files; expected ${EXPECTED_BY_DIR}`);
}

const manifestFiles = new Set(PREVIEW_GALLERY_MANIFEST.map((clip) => basename(clip.localPath)));
for (const file of [...tpvFiles, ...fpvFiles]) {
  if (!manifestFiles.has(file)) fail(`local file is not in manifest: ${file}`);
}

console.log(`[gallery] ok: ${PREVIEW_GALLERY_MANIFEST.length} clips mapped to preview/tpv + preview/fpv`);
