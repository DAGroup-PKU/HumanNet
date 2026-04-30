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
// If you add fields to SiteConfigBase / StoredSiteConfig
// (src/lib/config-types.ts), add them here too AND update
// src/lib/config-defaults.ts. Lesson L0013 covers keeping these in sync.

import { ACCENT_OPEN, ACCENT_CLOSE } from "./shared.js";

export const DEFAULT_CONFIG_PAYLOAD = {
  hero: {
    eyebrow: "Open Source · Embodied AI Dataset · v0.9 preview",
    title: `The data substrate for ${ACCENT_OPEN}embodied intelligence${ACCENT_CLOSE} that survives contact with the real world.`,
    description:
      "Project Nebula publishes a fully open, multi-modal capture of how humans and robots actually move, manipulate, and assemble — with synchronised exocentric and egocentric streams, raw enough to train on, structured enough to evaluate against.",
    primaryVideo: {
      kind: "local" as const,
      path: "/videos/exo/QSuxYRr3n7o_85.mp4",
    },
    primaryVideoAspect: 2134 / 1280,
    metrics: [
      { key: "Hours", value: "12.8k" },
      { key: "Tasks", value: "230" },
      { key: "Robots", value: "37" },
    ],
  },
  links: {
    github: "https://github.com/DAGroup-PKU",
    huggingface: "https://huggingface.co/DAGroup-PKU",
    waitlist: "https://tally.so/r/project-nebula-waitlist",
    discord: "#",
    mailingList: "#",
    codeOfConduct: "#",
  },
  team: [
    {
      id: "lh",
      name: "Linhua Hou",
      initials: "LH",
      role: "Project Lead",
      org: "Nebula Lab",
      focus: "Long-horizon embodied policies · dataset architecture",
    },
    {
      id: "ms",
      name: "Mira Saito",
      initials: "MS",
      role: "Capture Systems",
      org: "Nebula Lab",
      focus: "Multi-camera rig design · hardware synchronisation",
    },
    {
      id: "ya",
      name: "Yusuf Ahmadi",
      initials: "YA",
      role: "Annotation Tooling",
      org: "Nebula Lab",
      focus: "Six-panel renderer · skeleton & hand-pose pipelines",
    },
    {
      id: "rk",
      name: "Ravi Krishnan",
      initials: "RK",
      role: "Benchmark & Eval",
      org: "Sim-to-Real Group",
      focus: "Cross-embodiment evaluation · leaderboard infra",
    },
    {
      id: "es",
      name: "Elena Sokolova",
      initials: "ES",
      role: "Robotics Liaison",
      org: "Field Robotics Co-op",
      focus: "Factory & home capture deployments",
    },
    {
      id: "kj",
      name: "Kim Jiwoo",
      initials: "KJ",
      role: "Research Engineer",
      org: "Nebula Lab",
      focus: "Baseline VLA models · fine-tuning recipes",
    },
  ],
  gallery: [
    {
      id: "exo-manip",
      src: { kind: "local" as const, path: "/videos/exo/QSuxYRr3n7o_28.mp4" },
      perspective: "exo" as const,
      task: "Manipulation",
      caption: "Bimanual pick-and-place under a fixed studio rig.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "exo-nav",
      src: { kind: "local" as const, path: "/videos/exo/QSuxYRr3n7o_68.mp4" },
      perspective: "exo" as const,
      task: "Navigation",
      caption: "Cluttered indoor traversal with dynamic obstacles.",
      aspectRatio: 2134 / 1300,
    },
    {
      id: "exo-assembly",
      src: { kind: "local" as const, path: "/videos/exo/QSuxYRr3n7o_80.mp4" },
      perspective: "exo" as const,
      task: "Assembly",
      caption: "Multi-step part-fitting on a workbench.",
      aspectRatio: 2134 / 1322,
    },
    {
      id: "exo-tool",
      src: { kind: "local" as const, path: "/videos/exo/QSuxYRr3n7o_101.mp4" },
      perspective: "exo" as const,
      task: "Tool use",
      caption: "Hand-tool sequencing under occlusion.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "ego-factory",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0006_0.4284_factory_005_worker_011_0069__0_176_177_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "Factory work",
      caption: "Six-panel ego sweep on a production line.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-switch",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "Switch handling",
      caption: "First-person panel-switch operation, daylight rig.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-dslr",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0003_0.4354_z088-july-07-22-dslr__498_873_376_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "DSLR rig",
      caption: "Outdoor head-mounted DSLR capture.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-gopro",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0004_0.3625_z140-aug-16-22-gopro__855_1012_158_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "GoPro rig",
      caption: "GoPro-mounted egocentric workshop capture.",
      aspectRatio: 16 / 9,
    },
    {
      id: "exo-mobile",
      src: {
        kind: "local" as const,
        path: "/videos/exo/GQaByBr_QO0_64.mp4",
      },
      perspective: "exo" as const,
      task: "Mobile rig",
      caption: "Hand-held gimbal capture under mixed indoor lighting.",
      aspectRatio: 2134 / 1216,
    },
    {
      id: "ego-archive",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0001_0.4390_fbe0a401-9bf3-4725-bddb-b2c44472cda0__4312_4473_162_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "Archival run",
      caption: "Helmet-rig archival sequence with multi-camera sync.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-bench",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0001_0.5579_93414__33_191_159_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "Bench take",
      caption: "Numbered ego take from the open-bench capture set.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-workshop",
      src: {
        kind: "local" as const,
        path: "/videos/ego/0008_0.4179_25123__47_129_83_six_panel.mp4",
      },
      perspective: "ego" as const,
      task: "Workshop sweep",
      caption: "Workshop ego sweep with tool-set close-ups.",
      aspectRatio: 16 / 9,
    },
  ],
  footer: {
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
  },
};

export type DefaultConfig = typeof DEFAULT_CONFIG_PAYLOAD;
