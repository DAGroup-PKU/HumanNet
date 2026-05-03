// Bundled defaults in the *stored* shape used as the admin SPA's
// initial state before the API responds. Mirrors the backend seed
// (server/src/default-config.ts) for the brand-level fields (hero,
// footer, links). The gallery here is intentionally a single-clip
// placeholder — the real 40-clip Preview gallery only matters once
// the admin SPA has fetched live data via GET /api/admin/config.

import { ACCENT_OPEN, ACCENT_CLOSE } from "../lib/config-types";
import type { StoredSiteConfig } from "../lib/config-types";
import { DEFAULT_LINKS } from "../lib/links";
import { DEFAULT_FOOTER } from "../lib/config-defaults";

export const ADMIN_DEFAULT_CONFIG: StoredSiteConfig = {
  hero: {
    eyebrow: "Embodied AI · Human-Centric Dataset · Preview",
    title: `Scaling ${ACCENT_OPEN}Human-centric Video Learning${ACCENT_CLOSE} to One Million Hours`,
    description:
      "HumanNet aims to provide a comprehensive human-centric dataset, designed as scalable data infrastructure for training, evaluating, and advancing embodied learning models.",
    primaryVideo: { kind: "local", path: "/videos/exo/QSuxYRr3n7o_85.mp4" },
    primaryVideoAspect: 2134 / 1280,
    metrics: [
      { key: "Hours", value: "967k" },
      { key: "Scenes", value: "30+" },
      { key: "Tasks", value: "720k+" },
    ],
  },
  links: { ...DEFAULT_LINKS },
  team: [
    {
      id: "humannet-core",
      name: "HumanNet Core Team",
      initials: "HN",
      role: "Maintainers",
      org: "PKU DAGroup · SimpleSilicon",
      focus: "Per-person credits land alongside the methods paper.",
    },
  ],
  gallery: [
    {
      id: "fpv-ego-01",
      src: {
        kind: "oss",
        key: "preview/fpv/0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4",
      },
      perspective: "ego",
      task: "Switch handling",
      caption: "First-person panel-switch operation, daylight rig.",
      aspectRatio: 16 / 9,
    },
  ],
  footer: DEFAULT_FOOTER,
};
