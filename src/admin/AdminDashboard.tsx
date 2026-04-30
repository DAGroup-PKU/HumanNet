import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@heroui/react";
import { adminApi, HttpError, setToken } from "./api";
import type { SiteConfig } from "../lib/config-types";
import { DEFAULT_CONFIG } from "../lib/config-defaults";
import { HeroEditor } from "./editors/HeroEditor";
import { LinksEditor } from "./editors/LinksEditor";
import { TeamEditor } from "./editors/TeamEditor";
import { GalleryEditor } from "./editors/GalleryEditor";

type Tab = "hero" | "links" | "team" | "gallery";

const TABS: { id: Tab; label: string }[] = [
  { id: "hero", label: "Hero" },
  { id: "links", label: "Links" },
  { id: "team", label: "Team" },
  { id: "gallery", label: "Gallery" },
];

export function AdminDashboard() {
  const navigate = useNavigate();
  const [config, setConfig] = useState<SiteConfig>(DEFAULT_CONFIG);
  const [draft, setDraft] = useState<SiteConfig>(DEFAULT_CONFIG);
  const [tab, setTab] = useState<Tab>("hero");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const cfg = await adminApi.getConfig();
        setConfig(cfg);
        setDraft(cfg);
      } catch (e) {
        if (e instanceof HttpError && e.status === 401) {
          setToken(null);
          navigate("/admin/login", { replace: true });
          return;
        }
        setStatus({ kind: "err", msg: "Could not load config." });
      } finally {
        setLoading(false);
      }
    })();
  }, [navigate]);

  async function onSave() {
    setSaving(true);
    setStatus(null);
    try {
      const fresh = await adminApi.updateConfig(draft);
      setConfig(fresh);
      setDraft(fresh);
      setStatus({ kind: "ok", msg: `Saved · ${new Date().toLocaleTimeString()}` });
    } catch (e) {
      if (e instanceof HttpError) {
        if (e.status === 401) {
          setToken(null);
          navigate("/admin/login", { replace: true });
          return;
        }
        const detail =
          typeof e.payload === "object" && e.payload && "issues" in e.payload
            ? JSON.stringify(
                (e.payload as { issues: { path: string; message: string }[] })
                  .issues,
              )
            : `HTTP ${e.status}`;
        setStatus({ kind: "err", msg: `Save failed · ${detail}` });
      } else {
        setStatus({ kind: "err", msg: "Network error during save." });
      }
    } finally {
      setSaving(false);
    }
  }

  function onLogout() {
    setToken(null);
    navigate("/admin/login", { replace: true });
  }

  function onReset() {
    setDraft(config);
    setStatus({ kind: "ok", msg: "Draft reverted to last saved." });
  }

  const dirty = JSON.stringify(draft) !== JSON.stringify(config);

  if (loading) {
    return (
      <div className="grid min-h-dvh place-items-center font-mono text-xs uppercase tracking-[0.22em] text-nebula-on-dim">
        → Loading config …
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-h-dvh max-w-6xl flex-col px-6 py-8 sm:px-10 lg:px-12">
      <header className="flex flex-wrap items-center gap-4 border-b border-nebula-line pb-6">
        <div className="flex items-center gap-3">
          <span
            aria-hidden="true"
            className="grid h-8 w-8 place-items-center rounded-sm bg-nebula-primary/10 ring-1 ring-nebula-primary/40"
          >
            <span className="block h-2.5 w-2.5 rounded-full bg-nebula-primary" />
          </span>
          <div>
            <div className="font-display text-base text-nebula-on">
              Project <span className="text-nebula-primary">Nebula</span>
              <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                · admin
              </span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
              last saved · {config.updatedAt ?? "never"}
            </div>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Link
            to="/"
            className="rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
          >
            View site →
          </Link>
          <button
            type="button"
            onClick={onLogout}
            className="rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
          >
            Sign out
          </button>
        </div>
      </header>

      <nav
        aria-label="Editor sections"
        className="mt-6 flex flex-wrap items-center gap-2 border-b border-nebula-line pb-4"
      >
        {TABS.map((t) => {
          const active = t.id === tab;
          return (
            <button
              key={t.id}
              type="button"
              aria-pressed={active}
              onClick={() => setTab(t.id)}
              className={`rounded-sm border px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                active
                  ? "border-nebula-primary/60 bg-nebula-primary/15 text-nebula-on"
                  : "border-nebula-line bg-nebula-surface text-nebula-on-muted hover:border-nebula-line-strong hover:text-nebula-on"
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </nav>

      <main className="mt-6 flex-1 pb-32">
        {tab === "hero" && (
          <HeroEditor
            value={draft.hero}
            onChange={(hero) => setDraft({ ...draft, hero })}
          />
        )}
        {tab === "links" && (
          <LinksEditor
            value={draft.links}
            onChange={(links) => setDraft({ ...draft, links })}
          />
        )}
        {tab === "team" && (
          <TeamEditor
            value={draft.team}
            onChange={(team) => setDraft({ ...draft, team })}
          />
        )}
        {tab === "gallery" && (
          <GalleryEditor
            value={draft.gallery}
            onChange={(gallery) => setDraft({ ...draft, gallery })}
          />
        )}
      </main>

      <footer className="sticky bottom-0 -mx-6 mt-auto border-t border-nebula-line bg-nebula-base/95 px-6 py-4 backdrop-blur-md sm:-mx-10 sm:px-10 lg:-mx-12 lg:px-12">
        <div className="flex flex-wrap items-center gap-4">
          <span
            role="status"
            aria-live="polite"
            className={`font-mono text-[11px] uppercase tracking-[0.18em] ${
              status?.kind === "ok"
                ? "text-nebula-success"
                : status?.kind === "err"
                  ? "text-nebula-danger"
                  : "text-nebula-on-dim"
            }`}
          >
            {status?.msg ?? (dirty ? "→ Unsaved changes" : "→ All synced")}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={onReset}
              disabled={!dirty || saving}
              className="rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-nebula-on-muted disabled:opacity-40 hover:text-nebula-on"
            >
              Discard
            </button>
            <Button
              type="button"
              onClick={onSave}
              isDisabled={!dirty || saving}
              className="font-mono text-xs uppercase tracking-[0.18em]"
            >
              {saving ? "Saving…" : "Save changes"}
            </Button>
          </div>
        </div>
      </footer>
    </div>
  );
}
