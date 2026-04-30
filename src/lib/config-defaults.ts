import { MEMBERS } from "../data/members";
import { GALLERY } from "../data/gallery";
import { DEFAULT_LINKS } from "./links";
import { ACCENT_CLOSE, ACCENT_OPEN, type SiteConfig } from "./config-types";

// Bundled defaults — used (a) at first paint before the API responds and
// (b) as a permanent fallback when the API is unreachable (e.g. static
// preview, offline build). Keep these in sync with the seed in the
// backend (server/src/seed.ts) so a fresh DB matches the bundle.
export const DEFAULT_CONFIG: SiteConfig = {
  hero: {
    eyebrow: "Open Source · Embodied AI Dataset · v0.9 preview",
    title: `The data substrate for ${ACCENT_OPEN}embodied intelligence${ACCENT_CLOSE} that survives contact with the real world.`,
    description:
      "Project Nebula publishes a fully open, multi-modal capture of how humans and robots actually move, manipulate, and assemble — with synchronised exocentric and egocentric streams, raw enough to train on, structured enough to evaluate against.",
    primaryVideo: "/videos/exo/QSuxYRr3n7o_85.mp4",
    primaryVideoAspect: 2134 / 1280,
    metrics: [
      { key: "Hours", value: "12.8k" },
      { key: "Tasks", value: "230" },
      { key: "Robots", value: "37" },
    ],
  },
  links: { ...DEFAULT_LINKS },
  team: [...MEMBERS],
  gallery: [...GALLERY],
};
