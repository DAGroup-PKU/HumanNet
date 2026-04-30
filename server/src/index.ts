import express from "express";
import cors from "cors";
import { ENV } from "./env.js";
import { configRouter } from "./routes/config.js";
import { authRouter } from "./routes/auth.js";

const app = express();

// CORS: only the configured origin (defaults to the Vite dev server). In
// production where the SPA is served from the same Express instance, the
// browser doesn't issue preflights for the same-origin /api calls.
app.use(
  cors({
    origin: ENV.CORS_ORIGIN.split(",").map((s) => s.trim()),
    credentials: false,
  }),
);
app.use(express.json({ limit: "1mb" }));

// Trust the first proxy hop in case we're behind nginx — required for
// express-rate-limit to read the real client IP.
app.set("trust proxy", 1);

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", uptimeSeconds: Math.round(process.uptime()) });
});

app.use(authRouter);
app.use(configRouter);

app.use((_req, res) => {
  res.status(404).json({ error: "not_found" });
});

// Generic error handler — keeps the surface tight, never leaks the stack
// to the client. The detailed log goes to stderr for the operator.
app.use(
  (
    err: unknown,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction,
  ) => {
    // eslint-disable-next-line no-console
    console.error("[server] unhandled", err);
    res.status(500).json({ error: "internal_error" });
  },
);

app.listen(ENV.PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`[server] listening on http://localhost:${ENV.PORT}`);
});
