import type { StoredSiteConfig } from "../lib/config-types";

const TOKEN_KEY = "nebula:admin-token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(t: string | null) {
  try {
    if (t === null) localStorage.removeItem(TOKEN_KEY);
    else localStorage.setItem(TOKEN_KEY, t);
  } catch {
    /* ignore quota errors */
  }
}

export class HttpError extends Error {
  constructor(
    public status: number,
    public payload: unknown,
  ) {
    super(`HTTP ${status}`);
    this.name = "HttpError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { auth?: boolean } = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (init.auth) {
    const token = getToken();
    if (!token) throw new HttpError(401, { error: "no_token" });
    headers.set("Authorization", `Bearer ${token}`);
  }
  const res = await fetch(path, { ...init, headers });
  const text = await res.text();
  let payload: unknown = text;
  try {
    payload = text.length > 0 ? JSON.parse(text) : null;
  } catch {
    /* leave as text */
  }
  if (!res.ok) throw new HttpError(res.status, payload);
  return payload as T;
}

export interface LoginResponse {
  token: string;
  user: { id: number; username: string; role: "admin" };
}

export interface MeResponse {
  authenticated: boolean;
  user?: { id: number; username: string; role: "admin" };
}

export const adminApi = {
  login: (username: string, password: string) =>
    request<LoginResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  me: () =>
    request<MeResponse>("/api/auth/me", {
      auth: false,
      headers: getToken() ? { Authorization: `Bearer ${getToken()}` } : {},
    }),

  /** Returns the *raw* stored config — videos as discriminated unions
   *  (kind:"local"|"external"|"oss"), footer hrefs unresolved (still
   *  contain `$github` references). */
  getConfig: () => request<StoredSiteConfig>("/api/admin/config", { auth: true }),

  updateConfig: (next: StoredSiteConfig) =>
    request<StoredSiteConfig>("/api/admin/config", {
      method: "PUT",
      body: JSON.stringify(next),
      auth: true,
    }),
};
