// Single source-of-truth for the JSON shape exchanged between the public
// website and the backend admin service. Keep this file dependency-free
// so the same `.ts` can be imported by the Express server (via tsx).

import type { Member } from "../data/members";
import type { PerspectiveId } from "../data/perspectives";

// ─────────────────────────────────────────────────────────────────────
// Video references
// ─────────────────────────────────────────────────────────────────────
//
// Two levels:
//   - StoredVideo  → discriminated union written to the DB / edited in
//                    the admin UI. Never reaches the public site.
//   - RenderedVideo → a plain string URL the <video> element can load.
//                    The backend converts StoredVideo → RenderedVideo on
//                    every public GET (signing OSS keys at request time).
//
// This split is what lets the admin store a private OSS bucket+key while
// the public response only ever contains short-TTL signed URLs.

/** Stored on disk + sent to the admin SPA. Discriminated by `kind`. */
export type StoredVideo =
  /** A bundled local file under /public (e.g. /videos/exo/foo.mp4). */
  | { kind: "local"; path: string }
  /** A direct, public absolute URL (CDN, public OSS, etc.). */
  | { kind: "external"; url: string }
  /** A private Aliyun OSS object referenced by bucket + key. The server
   *  signs it (asyncSignatureUrl, expires=OSS_URL_TTL_SECONDS) before
   *  returning it to the public site. `bucket` is optional - when omitted
   *  the server falls back to OSS_BUCKET. */
  | { kind: "oss"; bucket?: string; key: string };

/** Sent to the public site - just a URL the browser can load. */
export type RenderedVideo = string;

// ─────────────────────────────────────────────────────────────────────
// Hero
// ─────────────────────────────────────────────────────────────────────

export interface HeroMetric {
  key: string;
  value: string;
}

export interface HeroConfigBase {
  eyebrow: string;
  /** Plain-text title. The accent fragment is wrapped between the
   *  ⟦ and ⟧ markers so the renderer can paint it in the brand colour
   *  without inviting markdown / HTML injection. */
  title: string;
  description: string;
  primaryVideoAspect: number;
  metrics: HeroMetric[];
}

export interface StoredHeroConfig extends HeroConfigBase {
  primaryVideo: StoredVideo;
}

export interface RenderedHeroConfig extends HeroConfigBase {
  primaryVideo: RenderedVideo;
}

// Backwards-compat alias - existing components consume the rendered shape.
export type HeroConfig = RenderedHeroConfig;

// ─────────────────────────────────────────────────────────────────────
// Gallery
// ─────────────────────────────────────────────────────────────────────

export interface GalleryClipBase {
  id: string;
  perspective: PerspectiveId;
  task: string;
  caption: string;
  /** native aspect of the asset (w / h), measured with ffprobe */
  aspectRatio: number;
}

export interface StoredGalleryClip extends GalleryClipBase {
  src: StoredVideo;
}

export interface RenderedGalleryClip extends GalleryClipBase {
  src: RenderedVideo;
}

// ─────────────────────────────────────────────────────────────────────
// Links
// ─────────────────────────────────────────────────────────────────────

export interface SiteLinks {
  github: string;
  huggingface: string;
  waitlist: string;
  discord: string;
  mailingList: string;
  codeOfConduct: string;
}

// ─────────────────────────────────────────────────────────────────────
// Footer
// ─────────────────────────────────────────────────────────────────────

/** A footer link item. `href` accepts:
 *   - an absolute URL ("https://..."),
 *   - a hash anchor ("#" or "#intro"),
 *   - a SiteLinks reference using the `$` prefix ("$github" → resolves
 *     to siteConfig.links.github at render time). The `$` form keeps the
 *     admin from having to update the same URL in two places.
 */
export interface FooterLink {
  label: string;
  href: string;
  external?: boolean;
}

export interface FooterColumn {
  id: string;
  title: string;
  items: FooterLink[];
}

export interface FooterLicenseTag {
  label: string;
}

export interface FooterConfig {
  /** The paragraph under the brand mark (currently the CC-BY-SA blurb). */
  brandTagline: string;
  /** Small chips beneath the tagline. Each is a free-form label. */
  licenses: FooterLicenseTag[];
  /** Right-hand columns. The renderer lays them out in a 3-column grid
   *  on lg+ screens regardless of count, so 2-4 columns work well. */
  columns: FooterColumn[];
  /** Bottom-row left text. */
  copyright: string;
  /** Bottom-row right text (version / status badge). */
  versionTag: string;
}

// ─────────────────────────────────────────────────────────────────────
// Top-level config: stored vs. rendered
// ─────────────────────────────────────────────────────────────────────

interface SiteConfigBase {
  links: SiteLinks;
  team: Member[];
  footer: FooterConfig;
  /** Set by the backend on each response so a stale cache can be detected. */
  updatedAt?: string;
}

/** Shape stored in the DB and edited by the admin SPA. */
export interface StoredSiteConfig extends SiteConfigBase {
  hero: StoredHeroConfig;
  gallery: StoredGalleryClip[];
}

/** Shape served to the public site (videos already rendered to URLs,
 *  footer hrefs already resolved). */
export interface SiteConfig extends SiteConfigBase {
  hero: RenderedHeroConfig;
  gallery: RenderedGalleryClip[];
}

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/**
 * Marker used inside `hero.title` to wrap the orange-accent fragment.
 * Example: "The data substrate for ⟦embodied intelligence⟧ that …"
 */
export const ACCENT_OPEN = "⟦";
export const ACCENT_CLOSE = "⟧";
