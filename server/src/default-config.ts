// Backend mirror of the frontend's bundled DEFAULT_CONFIG. We keep this
// stand-alone (rather than importing src/lib/config-defaults.ts) because
// (1) the frontend file imports React-aware modules, and (2) the seed
// script runs before any frontend tooling is involved.
//
// IMPORTANT — shape difference vs. the frontend default:
//   - Frontend's DEFAULT_CONFIG is the *rendered* shape (videos = string).
//   - This file is the *stored* shape (videos = StoredVideo discriminated
//     union). The server renders private OSS references as `/api/clip/*`
//     proxy URLs on every public GET — see server/src/render.ts.
//
// HumanNet Preview release notes (2026-Q2):
//   - Hero / footer / metrics / team / links are all the live HumanNet
//     payload (not the legacy "Project Nebula" placeholders).
//   - Gallery is 40 OSS-backed clips uploaded under preview/tpv and
//     preview/fpv in ss-oss-sites. Bucket defaults to ENV.OSS_BUCKET, so
//     the `bucket` field is omitted on each entry.
//   - Update server/src/preview-gallery.ts and run
//     `npm --prefix server run check:gallery` when the curated clip set
//     changes.
//
// If you add fields to SiteConfigBase / StoredSiteConfig
// (src/lib/config-types.ts), add them here too AND update
// src/lib/config-defaults.ts. Lesson L0013 covers keeping these in sync.

import { ACCENT_OPEN, ACCENT_CLOSE } from "./shared.js";
import { PREVIEW_GALLERY } from "./preview-gallery.js";

export const DEFAULT_CONFIG_PAYLOAD = {
  hero: {
    eyebrow: "Embodied AI · Human-Centric Dataset · Preview",
    title: `Scaling ${ACCENT_OPEN}Human-centric Video Learning${ACCENT_CLOSE} to One Million Hours`,
    description:
      "HumanNet aims to provide a comprehensive human-centric dataset, designed as scalable data infrastructure for training, evaluating, and advancing embodied learning models.",
    primaryVideo: {
      kind: "local" as const,
      path: "/videos/exo/QSuxYRr3n7o_85.mp4",
    },
    primaryVideoAspect: 2134 / 1280,
    metrics: [
      { key: "Hours", value: "967k" },
      { key: "Scenes", value: "30+" },
      { key: "Tasks", value: "720k+" },
    ],
  },
  links: {
    github: "#",
    huggingface: "#",
    waitlist: "https://tally.so/r/humannet-waitlist",
    discord: "#",
    mailingList: "#",
    codeOfConduct: "#",
    arxiv: "#",
  },
  // The public Members section now renders two organisations
  // (PKU DAGroup + SimpleSilicon) hard-coded in src/components/Members.tsx,
  // so this `team` array is currently unused on the visitor side. We
  // keep a small placeholder list to keep the admin tab functional and
  // to satisfy the `Member.min(?)` schema.
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
  gallery: PREVIEW_GALLERY,
  footer: {
    brandTagline:
      "An open, human-centric video corpus for embodied AI research. Released as a Preview build under CC-BY-SA 4.0 (data) and Apache 2.0 (code) by PKU DAGroup and SimpleSilicon.",
    licenses: [
      { label: "CC-BY-SA 4.0 · data" },
      { label: "Apache 2.0 · code" },
    ],
    // Preview release surfaces the two live external destinations
    // (Tally waitlist + arXiv). Items use the `$key` form so the URL is
    // edited once in the Links tab. Schema accepts up to 6 columns —
    // add more via the admin Footer tab as channels come online.
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
  },
};

export type DefaultConfig = typeof DEFAULT_CONFIG_PAYLOAD;
