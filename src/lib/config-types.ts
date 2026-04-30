// Single source-of-truth for the JSON shape exchanged between the public
// website and the backend admin service. Keep this file dependency-free
// so the same `.ts` can be imported by the Express server (via tsx).

import type { Member } from "../data/members";
import type { GalleryClip } from "../data/gallery";

export interface HeroMetric {
  key: string;
  value: string;
}

export interface HeroConfig {
  eyebrow: string;
  /** Plain-text title. The accent fragment is wrapped between the
   *  ⟦ and ⟧ markers so the renderer can paint it in the brand colour
   *  without inviting markdown / HTML injection. */
  title: string;
  description: string;
  /** Either an absolute URL (OSS / CDN) or a relative path under /public. */
  primaryVideo: string;
  primaryVideoAspect: number;
  metrics: HeroMetric[];
}

export interface SiteLinks {
  github: string;
  huggingface: string;
  waitlist: string;
  discord: string;
  mailingList: string;
  codeOfConduct: string;
}

export interface SiteConfig {
  hero: HeroConfig;
  links: SiteLinks;
  team: Member[];
  gallery: GalleryClip[];
  /** Set by the backend on each response so a stale cache can be detected. */
  updatedAt?: string;
}

/**
 * Marker used inside `hero.title` to wrap the orange-accent fragment.
 * Example: "The data substrate for ⟦embodied intelligence⟧ that …"
 */
export const ACCENT_OPEN = "⟦";
export const ACCENT_CLOSE = "⟧";
