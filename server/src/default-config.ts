// Backend mirror of the frontend's bundled DEFAULT_CONFIG. We keep this
// stand-alone (rather than importing src/lib/config-defaults.ts) because
// (1) the frontend file imports React-aware modules, and (2) the seed
// script runs before any frontend tooling is involved.
//
// If you add fields to SiteConfig (src/lib/config-types.ts), add them
// here too AND update src/lib/config-defaults.ts. Lesson L0013 covers
// keeping these in sync.

import { ACCENT_OPEN, ACCENT_CLOSE } from "./shared.js";

export const DEFAULT_CONFIG_PAYLOAD = {
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
      src: "/videos/exo/QSuxYRr3n7o_28.mp4",
      perspective: "exo",
      task: "Manipulation",
      caption: "Bimanual pick-and-place under a fixed studio rig.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "exo-nav",
      src: "/videos/exo/QSuxYRr3n7o_68.mp4",
      perspective: "exo",
      task: "Navigation",
      caption: "Cluttered indoor traversal with dynamic obstacles.",
      aspectRatio: 2134 / 1300,
    },
    {
      id: "exo-assembly",
      src: "/videos/exo/QSuxYRr3n7o_80.mp4",
      perspective: "exo",
      task: "Assembly",
      caption: "Multi-step part-fitting on a workbench.",
      aspectRatio: 2134 / 1322,
    },
    {
      id: "exo-tool",
      src: "/videos/exo/QSuxYRr3n7o_101.mp4",
      perspective: "exo",
      task: "Tool use",
      caption: "Hand-tool sequencing under occlusion.",
      aspectRatio: 2134 / 1280,
    },
    {
      id: "ego-factory",
      src: "/videos/ego/0006_0.4284_factory_005_worker_011_0069__0_176_177_six_panel.mp4",
      perspective: "ego",
      task: "Factory work",
      caption: "Six-panel ego sweep on a production line.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-switch",
      src: "/videos/ego/0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4",
      perspective: "ego",
      task: "Switch handling",
      caption: "First-person panel-switch operation, daylight rig.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-dslr",
      src: "/videos/ego/0003_0.4354_z088-july-07-22-dslr__498_873_376_six_panel.mp4",
      perspective: "ego",
      task: "DSLR rig",
      caption: "Outdoor head-mounted DSLR capture.",
      aspectRatio: 16 / 9,
    },
    {
      id: "ego-gopro",
      src: "/videos/ego/0004_0.3625_z140-aug-16-22-gopro__855_1012_158_six_panel.mp4",
      perspective: "ego",
      task: "GoPro rig",
      caption: "GoPro-mounted egocentric workshop capture.",
      aspectRatio: 16 / 9,
    },
  ],
};
