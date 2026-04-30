import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const API_TARGET = process.env.VITE_API_TARGET || "http://localhost:5175";

export default defineConfig({
  plugins: [react(), tailwindcss()],
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
