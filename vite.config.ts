import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { cpSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Plugin } from "vite";

const API_TARGET = process.env.VITE_API_TARGET || "http://localhost:5175";

function copyDirectoryWithoutDotfiles(src: string, dest: string) {
  mkdirSync(dest, { recursive: true });
  for (const entry of readdirSync(src)) {
    if (entry.startsWith(".")) continue;
    const from = join(src, entry);
    const to = join(dest, entry);
    if (statSync(from).isDirectory()) {
      copyDirectoryWithoutDotfiles(from, to);
    } else {
      mkdirSync(dirname(to), { recursive: true });
      cpSync(from, to);
    }
  }
}

function copyCuratedMediaPlugin(): Plugin {
  return {
    name: "copy-curated-media",
    apply: "build",
    closeBundle() {
      const source = "media/curation/video";
      if (!existsSync(source)) return;
      copyDirectoryWithoutDotfiles(source, "dist/media/curation/video");
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), copyCuratedMediaPlugin()],
  server: {
    // 5173 is Vite's default; sticking to it keeps the OSS Referer
    // whitelist (which only needs to know the standard dev port) simple.
    port: 5173,
    host: true,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
});
