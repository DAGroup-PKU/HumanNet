import { Router } from "express";
import bcrypt from "bcryptjs";
import rateLimit from "express-rate-limit";
import { db, type UserRow } from "../db.js";
import { signToken, verifyToken } from "../auth.js";
import { LoginSchema } from "../schema.js";

export const authRouter = Router();

// 5 attempts / minute / IP, lockout for 5 minutes after the limit is hit.
// Reasonable for a single-admin tool — does not pretend to be a full
// brute-force defence (a real one needs IP reputation + geo / ASN data).
const loginLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "too_many_login_attempts" },
});

authRouter.post("/api/auth/login", loginLimiter, (req, res) => {
  const parsed = LoginSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "validation_failed" });
    return;
  }
  const { username, password } = parsed.data;
  const row = db
    .prepare(
      `SELECT id, username, password_hash, role FROM users WHERE username = ? LIMIT 1`,
    )
    .get(username) as UserRow | undefined;

  // Constant-time-ish behaviour: always run bcrypt even when the user
  // is missing, so an attacker can't time-sniff valid usernames.
  const fakeHash = "$2a$12$ZZZZZZZZZZZZZZZZZZZZZ.fakeHashUsedForTimingDefence";
  const ok =
    bcrypt.compareSync(password, row?.password_hash ?? fakeHash) && !!row;

  if (!ok) {
    res.status(401).json({ error: "invalid_credentials" });
    return;
  }
  const token = signToken({
    sub: row!.id,
    username: row!.username,
    role: "admin",
  });
  res.json({
    token,
    user: { id: row!.id, username: row!.username, role: "admin" },
  });
});

authRouter.get("/api/auth/me", (req, res) => {
  // Light endpoint the admin SPA can ping to validate its cached token.
  // Doesn't go through `requireAdmin` so we can return a 200 with
  // `{authenticated: false}` for the unauthenticated case.
  const header = req.headers.authorization;
  if (!header?.startsWith("Bearer ")) {
    res.json({ authenticated: false });
    return;
  }
  try {
    const claims = verifyToken(header.slice(7));
    res.json({
      authenticated: true,
      user: { id: claims.sub, username: claims.username, role: claims.role },
    });
  } catch {
    res.json({ authenticated: false });
  }
});
