// Convert a StoredSiteConfig (DB shape — videos as discriminated unions,
// footer hrefs that may reference SiteLinks via the `$key` form) into the
// PublicSiteConfig the website consumes (videos as plain string URLs,
// fully-resolved footer hrefs).
//
// OSS-backed videos are emitted as opaque proxy URLs of the form
// `/api/clip/<id>.mp4`. The actual bytes are fetched server-side by the
// clip-cache module (see server/src/routes/clip.ts) — the browser never
// sees the bucket / region / key / signature.

import type { SiteConfigInput } from "./schema.js";

/** Minimal public shape returned by GET /api/config. */
export interface PublicSiteConfig {
  hero: {
    eyebrow: string;
    title: string;
    description: string;
    primaryVideo: string;
    primaryVideoAspect: number;
    metrics: { key: string; value: string }[];
  };
  links: SiteConfigInput["links"];
  team: SiteConfigInput["team"];
  gallery: Array<{
    id: string;
    src: string;
    perspective: "exo" | "ego";
    task: string;
    caption: string;
    aspectRatio: number;
  }>;
  footer: {
    brandTagline: string;
    licenses: { label: string }[];
    columns: Array<{
      id: string;
      title: string;
      items: { label: string; href: string; external?: boolean }[];
    }>;
    copyright: string;
    versionTag: string;
  };
  updatedAt?: string;
}

type StoredVideo = SiteConfigInput["hero"]["primaryVideo"];

/** Turn a StoredVideo into a URL the `<video>` element can consume.
 *
 *  - kind:"local"    → returned unchanged (`/videos/...` served by Vite
 *                       in dev, by nginx static in prod).
 *  - kind:"external" → returned unchanged (any public CDN URL).
 *  - kind:"oss"      → opaque `/api/clip/<id>.mp4` proxy URL. The id
 *                       must match the entry's id in the stored config
 *                       (gallery `c.id`, or `hero-primary` for the hero
 *                       video). The clip route looks the id up to find
 *                       the bucket+key and serves bytes via OSS SDK. */
function renderVideo(v: StoredVideo, fallbackId: string): string {
  switch (v.kind) {
    case "local":
      return v.path;
    case "external":
      return v.url;
    case "oss":
      return `/api/clip/${fallbackId}.mp4`;
  }
}

/** Resolve a footer href.
 *   - "$key"  → siteConfig.links[key] (or the placeholder "#" when the
 *               key doesn't exist; we don't throw because the admin may
 *               temporarily point at a not-yet-defined key).
 *   - else    → returned unchanged.
 */
function resolveFooterHref(
  href: string,
  links: SiteConfigInput["links"],
): string {
  if (!href.startsWith("$")) return href;
  const key = href.slice(1) as keyof SiteConfigInput["links"];
  return links[key] ?? "#";
}

export function renderPublicConfig(
  stored: SiteConfigInput,
  updatedAt?: string,
): PublicSiteConfig {
  return {
    hero: {
      eyebrow: stored.hero.eyebrow,
      title: stored.hero.title,
      description: stored.hero.description,
      primaryVideo: renderVideo(stored.hero.primaryVideo, "hero-primary"),
      primaryVideoAspect: stored.hero.primaryVideoAspect,
      metrics: stored.hero.metrics,
    },
    links: stored.links,
    team: stored.team,
    gallery: stored.gallery.map((c) => ({
      id: c.id,
      src: renderVideo(c.src, c.id),
      perspective: c.perspective,
      task: c.task,
      caption: c.caption,
      aspectRatio: c.aspectRatio,
    })),
    footer: {
      brandTagline: stored.footer.brandTagline,
      licenses: stored.footer.licenses,
      columns: stored.footer.columns.map((col) => ({
        id: col.id,
        title: col.title,
        items: col.items.map((it) => ({
          label: it.label,
          href: resolveFooterHref(it.href, stored.links),
          external: it.external,
        })),
      })),
      copyright: stored.footer.copyright,
      versionTag: stored.footer.versionTag,
    },
    updatedAt,
  };
}
