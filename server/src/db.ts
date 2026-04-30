import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { ENV } from "./env.js";

mkdirSync(path.dirname(ENV.DB_PATH), { recursive: true });

export const db = new Database(ENV.DB_PATH);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");
db.pragma("synchronous = NORMAL");

// Schema: a single-row site_config (denormalised JSON blob — the admin
// edits the entire object atomically, so a row-per-section model would
// just add complexity for no win) plus a users table for auth.
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS site_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_by TEXT
  );
`);

export type ConfigRow = {
  id: 1;
  payload: string;
  updated_at: string;
  updated_by: string | null;
};

export type UserRow = {
  id: number;
  username: string;
  password_hash: string;
  role: string;
};
