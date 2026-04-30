import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { DEFAULT_CONFIG } from "./config-defaults";
import type { SiteConfig } from "./config-types";

interface ConfigContextValue {
  config: SiteConfig;
  /** True until the first API fetch resolves (success or failure). */
  loading: boolean;
  /** Last fetch error. `null` when bundled defaults are in use without
   *  the API being attempted yet. */
  error: string | null;
  /** Force a refetch — the admin UI calls this after a successful save. */
  reload: () => Promise<void>;
}

const ConfigContext = createContext<ConfigContextValue | null>(null);

const API_URL = "/api/config";

async function fetchConfig(): Promise<SiteConfig> {
  const res = await fetch(API_URL, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`GET ${API_URL} → HTTP ${res.status}`);
  const json = (await res.json()) as Partial<SiteConfig>;
  // Defensive merge: if the backend forgets a field (or sends a stale
  // schema version), we still render with the bundled default.
  return {
    hero: { ...DEFAULT_CONFIG.hero, ...(json.hero ?? {}) },
    links: { ...DEFAULT_CONFIG.links, ...(json.links ?? {}) },
    team: json.team ?? DEFAULT_CONFIG.team,
    gallery: json.gallery ?? DEFAULT_CONFIG.gallery,
    updatedAt: json.updatedAt,
  };
}

export function ConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<SiteConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const fresh = await fetchConfig();
      setConfig(fresh);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "unknown");
      // Keep DEFAULT_CONFIG (or last successful config) on screen.
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const value = useMemo<ConfigContextValue>(
    () => ({ config, loading, error, reload }),
    [config, loading, error, reload],
  );

  return (
    <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>
  );
}

/**
 * Convenience hook — returns the live `SiteConfig` plus its top-level
 * slices spread out for ergonomic destructuring.
 *
 * ```tsx
 * const { hero, team, links, loading } = useConfig();
 * ```
 */
export function useConfig() {
  const ctx = useContext(ConfigContext);
  if (!ctx) {
    // Permits component usage outside the provider during tests / Storybook.
    return {
      config: DEFAULT_CONFIG,
      hero: DEFAULT_CONFIG.hero,
      links: DEFAULT_CONFIG.links,
      team: DEFAULT_CONFIG.team,
      gallery: DEFAULT_CONFIG.gallery,
      loading: false,
      error: null,
      reload: async () => {},
    };
  }
  return {
    ...ctx,
    hero: ctx.config.hero,
    links: ctx.config.links,
    team: ctx.config.team,
    gallery: ctx.config.gallery,
  };
}
