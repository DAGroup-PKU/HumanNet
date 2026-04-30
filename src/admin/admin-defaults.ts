// Bundled defaults in the *stored* shape used as the admin SPA's
// initial state before the API responds. Mirrors the backend seed
// (server/src/default-config.ts) so editing a fresh DB shows the same
// content the visitor sees.

import { ACCENT_OPEN, ACCENT_CLOSE } from "../lib/config-types";
import type { StoredSiteConfig } from "../lib/config-types";
import { MEMBERS } from "../data/members";
import { DEFAULT_LINKS } from "../lib/links";
import { DEFAULT_FOOTER } from "../lib/config-defaults";

export const ADMIN_DEFAULT_CONFIG: StoredSiteConfig = {
  hero: {
    eyebrow: "Open Source · Embodied AI Dataset · v0.9 preview",
    title: `The data substrate for ${ACCENT_OPEN}embodied intelligence${ACCENT_CLOSE} that survives contact with the real world.`,
    description:
      "Project Nebula publishes a fully open, multi-modal capture of how humans and robots actually move, manipulate, and assemble — with synchronised exocentric and egocentric streams, raw enough to train on, structured enough to evaluate against.",
    primaryVideo: { kind: "local", path: "/videos/exo/QSuxYRr3n7o_85.mp4" },
    primaryVideoAspect: 2134 / 1280,
    metrics: [
      { key: "Hours", value: "12.8k" },
      { key: "Tasks", value: "230" },
      { key: "Robots", value: "37" },
    ],
  },
  links: { ...DEFAULT_LINKS },
  team: [...MEMBERS],
  gallery: [
    {
      id: "exo-manip",
      src: { kind: "local", path: "/videos/exo/QSuxYRr3n7o_28.mp4" },
      perspective: "exo",
      task: "Manipulation",
      caption: "Bimanual pick-and-place under a fixed studio rig.",
      aspectRatio: 2134 / 1280,
    },
  ],
  footer: DEFAULT_FOOTER,
};
