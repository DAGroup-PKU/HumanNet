import { MEMBERS } from "../data/members";
import { GALLERY } from "../data/gallery";
import { DEFAULT_LINKS } from "./links";
import {
  ACCENT_CLOSE,
  ACCENT_OPEN,
  type FooterConfig,
  type SiteConfig,
} from "./config-types";

// Bundled defaults — used (a) at first paint before the API responds and
// (b) as a permanent fallback when the API is unreachable (e.g. static
// preview, offline build). Keep these in sync with the seed in the
// backend (server/src/default-config.ts) so a fresh DB matches the bundle.
//
// Note: this is the *rendered* SiteConfig (videos as plain strings,
// footer hrefs already resolved). The backend stores a StoredSiteConfig
// and renders it on every public GET. See server/src/render.ts.

export const DEFAULT_FOOTER: FooterConfig = {
  brandTagline:
    "An open, human-centric video corpus for embodied AI research. Released as a Preview build under CC-BY-SA 4.0 (data) and Apache 2.0 (code) by PKU DAGroup and SimpleSilicon.",
  licenses: [
    { label: "CC-BY-SA 4.0 · data" },
    { label: "Apache 2.0 · code" },
  ],
  // The Preview release surfaces just the two live external destinations
  // (Tally waitlist + arXiv paper). The `$key` form resolves to
  // siteConfig.links.<key> at render time so the admin only edits the
  // URL in one place (Links tab). When more channels are ready, add
  // them via the admin Footer tab — schema accepts up to 6 columns.
  columns: [
    {
      id: "get-involved",
      title: "Get involved",
      items: [
        { label: "Join the waitlist", href: "$waitlist", external: true },
        { label: "Read the paper", href: "$arxiv", external: true },
      ],
    },
  ],
  copyright: "© 2026 PKU DAGroup · SimpleSilicon.",
  versionTag: "Preview Version",
};

export const DEFAULT_CONFIG: SiteConfig = {
  hero: {
    eyebrow: "Embodied AI · Human-Centric Dataset · Preview",
    title: `Scaling ${ACCENT_OPEN}Human-centric Video Learning${ACCENT_CLOSE} to One Million Hours`,
    description:
      "HumanNet aims to provide a comprehensive human-centric dataset, designed as scalable data infrastructure for training, evaluating, and advancing embodied learning models.",
    primaryVideo: "/videos/exo/QSuxYRr3n7o_85.mp4",
    primaryVideoAspect: 2134 / 1280,
    metrics: [
      { key: "Hours", value: "967k" },
      { key: "Scenes", value: "30+" },
      { key: "Tasks", value: "720k+" },
    ],
  },
  links: { ...DEFAULT_LINKS },
  team: [...MEMBERS],
  gallery: [...GALLERY],
  footer: DEFAULT_FOOTER,
};
