export type RoadmapStatus = "shipped" | "active" | "queued";

export interface RoadmapMilestone {
  quarter: string;
  title: string;
  status: RoadmapStatus;
  body: string;
  artefacts: string[];
}

// HumanNet Preview roadmap. Two near-term milestones; both unshipped.
// Keeping the list intentionally short — anything we haven't started yet
// belongs in a research log, not on the public site.
export const ROADMAP: RoadmapMilestone[] = [
  {
    quarter: "Q3 · 2026",
    title: "HumanNet Benchmark v1",
    status: "active",
    body: "An embodied-evaluation suite spanning third-person and egocentric perception, recognition, and downstream policy-tuning probes — released alongside reference baselines so external submissions are reproducible end-to-end.",
    artefacts: [
      "task taxonomy",
      "evaluation harness",
      "baseline weights",
    ],
  },
  {
    quarter: "Q4 · 2026",
    title: "Ego-data scaling-law study",
    status: "queued",
    body: "Empirical validation of how embodied policies scale as the egocentric subset grows from 100k → 747k hours. The goal is a published curve that lets practitioners decide which slice of HumanNet to download for a given compute budget.",
    artefacts: [
      "scaling-curve report",
      "subset manifests",
      "reproduction scripts",
    ],
  },
];
