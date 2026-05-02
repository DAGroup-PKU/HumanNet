import type { NextFunction, Request, Response } from "express";
import jwt from "jsonwebtoken";
import { ENV } from "./env.js";

export interface AdminClaims {
  sub: number; // user id
  username: string;
  role: "admin";
}

const ALGO: jwt.Algorithm = "HS256";

export function signToken(claims: AdminClaims): string {
  return jwt.sign(claims, ENV.JWT_SECRET, {
    algorithm: ALGO,
    expiresIn: ENV.JWT_TTL_SECONDS,
  });
}

export function verifyToken(token: string): AdminClaims {
  const decoded = jwt.verify(token, ENV.JWT_SECRET, {
    algorithms: [ALGO],
  }) as unknown as AdminClaims;
  if (
    !decoded ||
    typeof decoded !== "object" ||
    decoded.role !== "admin" ||
    typeof decoded.sub !== "number" ||
    typeof decoded.username !== "string"
  ) {
    throw new Error("Token claims are invalid");
  }
  return decoded;
}

declare module "express-serve-static-core" {
  interface Request {
    admin?: AdminClaims;
  }
}

export function requireAdmin(
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    res.status(401).json({ error: "missing_bearer_token" });
    return;
  }
  const token = header.slice("Bearer ".length).trim();
  try {
    req.admin = verifyToken(token);
    next();
  } catch (err) {
    res.status(401).json({
      error: "invalid_token",
      detail: err instanceof Error ? err.message : String(err),
    });
  }
}
