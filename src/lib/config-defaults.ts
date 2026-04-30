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
    "An open-source embodied AI dataset & toolchain. Released under CC-BY-SA 4.0 (data) and Apache 2.0 (code). Built by an open community of robotics researchers, engineers, and field operators.",
  licenses: [
    { label: "CC-BY-SA 4.0 · data" },
    { label: "Apache 2.0 · code" },
  ],
  columns: [
    {
      id: "project",
      title: "Project",
      items: [
        { label: "Dataset card", href: "#" },
        { label: "Loader SDK", href: "#" },
        { label: "Calibration tool", href: "#" },
        { label: "Benchmarks", href: "#" },
      ],
    },
    {
      id: "community",
      title: "Community",
      items: [
        { label: "GitHub", href: "$github", external: true },
        { label: "Hugging Face", href: "$huggingface", external: true },
        { label: "Discord", href: "$discord" },
        { label: "Mailing list", href: "$mailingList" },
      ],
    },
    {
      id: "research",
      title: "Research",
      items: [
        { label: "Methods paper", href: "#" },
        { label: "Capture rigs", href: "#" },
        { label: "Annotation schema", href: "#" },
        { label: "Reproducibility", href: "#" },
      ],
    },
  ],
  copyright: "© 2025–2026 Project Nebula contributors.",
  versionTag: "v0.9 · preview · embargo 2026-Q3",
};

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
  footer: DEFAULT_FOOTER,
};
