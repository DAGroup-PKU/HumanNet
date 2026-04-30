// Convert a StoredSiteConfig (DB shape — videos as discriminated unions,
// footer hrefs that may reference SiteLinks via the `$key` form) into the
// PublicSiteConfig the website consumes (videos as plain string URLs,
// fully-resolved footer hrefs).
//
// All this happens on every public GET /api/config so signed URLs stay
// fresh. The cost is a few ali-oss signature computations per request,
// which is local CPU only (no network round-trip), so it's negligible
// even if every clip ends up backed by OSS.

import { signOssRef } from "./oss.js";
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

async function renderVideo(v: StoredVideo): Promise<string> {
  switch (v.kind) {
    case "local":
      return v.path;
    case "external":
      return v.url;
    case "oss":
      return signOssRef({ bucket: v.bucket, key: v.key });
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

export async function renderPublicConfig(
  stored: SiteConfigInput,
  updatedAt?: string,
): Promise<PublicSiteConfig> {
  const heroVideoUrl = await renderVideo(stored.hero.primaryVideo);
  const galleryUrls = await Promise.all(
    stored.gallery.map((c) => renderVideo(c.src)),
  );

  return {
    hero: {
      eyebrow: stored.hero.eyebrow,
      title: stored.hero.title,
      description: stored.hero.description,
      primaryVideo: heroVideoUrl,
      primaryVideoAspect: stored.hero.primaryVideoAspect,
      metrics: stored.hero.metrics,
    },
    links: stored.links,
    team: stored.team,
    gallery: stored.gallery.map((c, i) => ({
      id: c.id,
      src: galleryUrls[i]!,
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
