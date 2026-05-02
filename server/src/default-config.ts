// Backend mirror of the frontend's bundled DEFAULT_CONFIG. We keep this
// stand-alone (rather than importing src/lib/config-defaults.ts) because
// (1) the frontend file imports React-aware modules, and (2) the seed
// script runs before any frontend tooling is involved.
//
// IMPORTANT — shape difference vs. the frontend default:
//   - Frontend's DEFAULT_CONFIG is the *rendered* shape (videos = string).
//   - This file is the *stored* shape (videos = StoredVideo discriminated
//     union). The server signs/resolves them on every public GET — see
//     server/src/render.ts. Bundled clip URLs default to kind:"local" so
//     they are served unchanged from the Vite /public/videos/ tree.
//
// HumanNet Preview release notes (2026-Q2):
//   - Hero / footer / metrics / team / links are all the live HumanNet
//     payload (not the legacy "Project Nebula" placeholders).
//   - Gallery is 40 OSS-backed clips: 20 third-person (10 Koala-36M-v1
//     + 10 OpenHumanVid) + 20 egocentric six-panel renderings. Bucket
//     defaults to ENV.OSS_BUCKET (= ss-oss-sites) so the `bucket` field
//     is omitted on each entry.
//   - Re-generate the gallery block via `npx tsx scripts/gen-gallery.ts`
//     when the curated clip set changes.
//
// If you add fields to SiteConfigBase / StoredSiteConfig
// (src/lib/config-types.ts), add them here too AND update
// src/lib/config-defaults.ts. Lesson L0013 covers keeping these in sync.

import { ACCENT_OPEN, ACCENT_CLOSE } from "./shared.js";

export const DEFAULT_CONFIG_PAYLOAD = {
  hero: {
    eyebrow: "Embodied AI · Human-Centric Dataset · Preview Version",
    title: `Scaling ${ACCENT_OPEN}Human-centric Video Learning${ACCENT_CLOSE} to One Million Hours`,
    description:
      "HumanNet aims to provide a comprehensive human-centric dataset, served as a scalable data infrastructure for training, evaluating, and advancing embodied learning models.",
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
  gallery: [
    // ──────────────────── BEGIN GENERATED ────────────────────
    {
      id: "tpv-koala-01",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0001/GQaByBr_QO0_64.mp4" },
      perspective: "exo" as const,
      task: "Mobile rig",
      caption: "Hand-held gimbal capture under mixed indoor lighting.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-02",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0001/HnjaKDRnmFM_64.mp4" },
      perspective: "exo" as const,
      task: "Studio rig",
      caption: "Multi-shot studio sequence from the Koala-36M montage subset.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-03",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0001/QSuxYRr3n7o_28.mp4" },
      perspective: "exo" as const,
      task: "Manipulation",
      caption: "Bimanual pick-and-place under a fixed studio rig.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-04",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0001/QSuxYRr3n7o_68.mp4" },
      perspective: "exo" as const,
      task: "Navigation",
      caption: "Cluttered indoor traversal with dynamic obstacles.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-05",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0001/QSuxYRr3n7o_80.mp4" },
      perspective: "exo" as const,
      task: "Assembly",
      caption: "Multi-step part-fitting on a workbench.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-06",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0001/QSuxYRr3n7o_101.mp4" },
      perspective: "exo" as const,
      task: "Tool use",
      caption: "Hand-tool sequencing under occlusion.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-07",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0006/KnMVTT5EcXM_14.mp4" },
      perspective: "exo" as const,
      task: "Workshop",
      caption: "Workshop activity recorded in a bench-top capture rig.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-08",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0007/WnJbxVE9jJE_43.mp4" },
      perspective: "exo" as const,
      task: "Studio rig",
      caption: "Multi-actor studio capture, mid-distance framing.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-09",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0009/9UkZ-mmDXAg_0.mp4" },
      perspective: "exo" as const,
      task: "Daily activity",
      caption: "Routine daily-life action under a fixed studio camera.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-koala-10",
      src: { kind: "oss" as const, key: "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/Koala-36M-v1_sub_001_part_0011/K4aoW7LCDK0_167.mp4" },
      perspective: "exo" as const,
      task: "Daily activity",
      caption: "Single-actor daily activity from the Koala-36M corpus.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "tpv-ohv-01",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0021/14486e0ba62338990040b634e052900b.mp4" },
      perspective: "exo" as const,
      task: "Crowd scene",
      caption: "Crowd-sourced human activity from the OpenHumanVid corpus.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-02",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0021/14b65a2bc1a17f9cce0f6abb9853d343.mp4" },
      perspective: "exo" as const,
      task: "Daily activity",
      caption: "Wide-shot daily activity, mid-scene framing.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-03",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0022/1523721c60d01bcf6b9461532008a00e.mp4" },
      perspective: "exo" as const,
      task: "Outdoor scene",
      caption: "Outdoor sequence from the OpenHumanVid set.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-04",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0022/1570bbe5e5eb9c0710abb9525bcef8ca.mp4" },
      perspective: "exo" as const,
      task: "Indoor scene",
      caption: "Indoor crowd-sourced clip, mid-distance framing.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-05",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0023/16908c584f98bc9a02526131ba9601fc.mp4" },
      perspective: "exo" as const,
      task: "Long take",
      caption: "Extended single-camera take, multi-actor.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-06",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0023/16e13b754e967eea1dc187875b789e1a.mp4" },
      perspective: "exo" as const,
      task: "Daily activity",
      caption: "Routine human activity, OpenHumanVid sample.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-07",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0024/175b2709b6a92b1e0556228ebd921b09.mp4" },
      perspective: "exo" as const,
      task: "Indoor scene",
      caption: "Indoor activity clip, OpenHumanVid sample.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-08",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0025/184fe60ca3642a57c95a1df666533550.mp4" },
      perspective: "exo" as const,
      task: "Wide shot",
      caption: "Wide-angle human-activity capture.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-09",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0025/18a3f9ba80e256ab15740ab1abff85a3.mp4" },
      perspective: "exo" as const,
      task: "Crowd scene",
      caption: "Multi-person scene from the OpenHumanVid corpus.",
      aspectRatio: 16 / 9,
    },
    {
      id: "tpv-ohv-10",
      src: { kind: "oss" as const, key: "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/OpenHumanVid_sub_001_part_0026/1992c6bdec3f8731069439bbbda15a25.mp4" },
      perspective: "exo" as const,
      task: "Indoor scene",
      caption: "Indoor session, OpenHumanVid sample.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-01",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Switch handling",
      caption: "First-person panel-switch operation, daylight rig.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-02",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0001_0.4390_fbe0a401-9bf3-4725-bddb-b2c44472cda0__4312_4473_162_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Archival run",
      caption: "Helmet-rig archival sequence with multi-camera sync.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-03",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0001_0.5579_93414__33_191_159_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Bench take",
      caption: "Numbered ego take from the open-bench capture set.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-04",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0003_0.4354_z088-july-07-22-dslr__498_873_376_six_panel.mp4" },
      perspective: "ego" as const,
      task: "DSLR rig",
      caption: "Outdoor head-mounted DSLR capture.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-05",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0004_0.3625_z140-aug-16-22-gopro__855_1012_158_six_panel.mp4" },
      perspective: "ego" as const,
      task: "GoPro rig",
      caption: "GoPro-mounted egocentric workshop capture.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-06",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0006_0.3873_63009__44_122_79_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Bench take",
      caption: "Egocentric bench take with six-panel rendering.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-07",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0006_0.4284_factory_005_worker_011_0069__0_176_177_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Factory work",
      caption: "Six-panel ego sweep on a production line.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-08",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0008_0.4179_25123__47_129_83_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Workshop sweep",
      caption: "Workshop ego sweep with tool-set close-ups.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-09",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0010_0.5277_z028-june-22-22-marius_assemble__5373_5505_133_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Assembly",
      caption: "Studio-rigged assembly task, ego perspective.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-10",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0011_0.3060_09f8cba3-0337-4043-b341-c241d5248cab__73800_74025_226_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Archival run",
      caption: "Long ego archival sequence, multi-camera sync.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-11",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0012_0.3559_z035-june-23-22-rashult_assemble__9188_9362_175_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Assembly",
      caption: "Studio-rig assembly task with hand-pose tracking.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-12",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0013_0.3327_R030-12July-Nespresso__3621_3794_174_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Coffee machine",
      caption: "Coffee-machine interaction, head-mounted ego rig.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-13",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0013_0.3507_factory011_worker035_00053__2315_2548_234_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Factory work",
      caption: "Production-line worker capture, ego six-panel.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-14",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0013_0.4577_factory_101_worker_013_0112__3407_3632_226_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Factory work",
      caption: "Factory bench task, ego perspective.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-15",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0017_0.3073_z132-aug-12-22-nespresso__2955_3120_166_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Coffee machine",
      caption: "Espresso preparation captured on a head-mounted rig.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-16",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0017_0.3760_factory053_worker007_00090__510_635_126_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Factory work",
      caption: "Worker-side production capture, ego six-panel.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-17",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0026_0.3115_z039-june-23-22-nespresso__5001_5261_261_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Coffee machine",
      caption: "Coffee-machine routine, head-mounted ego.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-18",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0028_0.3739_R086-27July-Nespresso__1193_1319_127_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Coffee machine",
      caption: "Repeat coffee-prep capture, ego six-panel.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-19",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0031_0.3146_z025-june-21-22-printer_small__1138_1386_249_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Office tool",
      caption: "Office-printer interaction from the ego studio rig.",
      aspectRatio: 16 / 9,
    },
    {
      id: "fpv-ego-20",
      src: { kind: "oss" as const, key: "data/dataset/egocentric/show_cases/selected/0050_0.3851_c2b3e7ea-fbeb-4491-b5bc-d0dd723496f7__62861_63006_146_six_panel.mp4" },
      perspective: "ego" as const,
      task: "Archival run",
      caption: "Long archival ego sequence with multi-camera sync.",
      aspectRatio: 16 / 9,
    },
    // ──────────────────── END GENERATED ────────────────────
  ],
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
