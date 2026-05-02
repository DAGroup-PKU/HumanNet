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
  // The Preview release intentionally hides the per-domain link
  // columns; HumanNet only exposes waitlist + arXiv to visitors today
  // and the Footer brand area is enough on its own. The columns array
  // can be re-populated via the admin tab once GitHub / HF / Discord
  // channels are ready.
  columns: [],
  copyright: "© 2026 PKU DAGroup · SimpleSilicon.",
  versionTag: "Preview Version",
};

export const DEFAULT_CONFIG: SiteConfig = {
  hero: {
    eyebrow: "Embodied AI · Human-Centric Dataset · Preview Version",
    title: `Scaling ${ACCENT_OPEN}Human-centric Video Learning${ACCENT_CLOSE} to One Million Hours`,
    description:
      "HumanNet aims to provide a comprehensive human-centric dataset, served as a scalable data infrastructure for training, evaluating, and advancing embodied learning models.",
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
