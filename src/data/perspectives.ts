export type PerspectiveId = "exo" | "ego";

export interface PerspectiveSpec {
  id: PerspectiveId;
  label: string;
  short: string;
  tagline: string;
  description: string;
  bullets: { label: string; value: string }[];
  /** Path under /public — must be served as /videos/... */
  primaryVideo: string;
  /** Native aspect ratio of the source clip (width / height). */
  aspectRatio: number;
}

export const PERSPECTIVES: PerspectiveSpec[] = [
  {
    id: "exo",
    label: "Exocentric",
    short: "Third-person",
    tagline: "Stage-anchored cameras observing whole-body manipulation, locomotion, and assembly tasks.",
    description:
      "Exocentric clips frame the agent from the outside—static rigs, hand-held gimbals, fixed industrial cameras. They preserve the geometry of the workspace and reveal long-horizon intent that ego-only views miss.",
    bullets: [
      { label: "Sources", value: "studio rigs · field gimbals · factory CCTV" },
      { label: "Native FoV", value: "wide (35–70°)" },
      { label: "Typical resolution", value: "1080p · 30 fps" },
      { label: "Annotation", value: "skeleton · object track · phase label" },
    ],
    primaryVideo: "/videos/exo/HnjaKDRnmFM_64.mp4",
    aspectRatio: 2134 / 1238,
  },
  {
    id: "ego",
    label: "Egocentric",
    short: "First-person",
    tagline: "Head-mounted, six-panel research clips synchronizing RGB, depth, hand pose, and rendered reference views.",
    description:
      "Egocentric clips are first-person captures from helmet/glasses rigs, exported as a six-panel research montage. The full panel layout is preserved here so researchers can verify alignment between RGB, depth, mesh, and hand-pose references.",
    bullets: [
      { label: "Sources", value: "Aria glasses · GoPro · custom helmet rig" },
      { label: "Native FoV", value: "ultra-wide (90°+)" },
      { label: "Typical resolution", value: "six-panel 1.5×3 grid · 30 fps" },
      { label: "Annotation", value: "RGB-D · gaze · 6-DoF hand pose" },
    ],
    primaryVideo: "/videos/ego/0006_0.3873_63009__44_122_79_six_panel.mp4",
    aspectRatio: 16 / 9,
  },
];
