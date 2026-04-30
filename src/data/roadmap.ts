export type RoadmapStatus = "shipped" | "active" | "queued";

export interface RoadmapMilestone {
  quarter: string;
  title: string;
  status: RoadmapStatus;
  body: string;
  artefacts: string[];
}

export const ROADMAP: RoadmapMilestone[] = [
  {
    quarter: "Q3 · 2025",
    title: "Nebula-1 corpus released",
    status: "shipped",
    body: "Public drop of 4,200 hours of synchronised exocentric + egocentric capture, with skeletons and phase labels.",
    artefacts: ["dataset card", "loader sdk", "baseline weights"],
  },
  {
    quarter: "Q4 · 2025",
    title: "Six-panel ego pipeline open-sourced",
    status: "shipped",
    body: "Reference implementation of the RGB / depth / hand-pose / mesh six-panel renderer used by the ego subset.",
    artefacts: ["panel-renderer", "calibration tool", "qa harness"],
  },
  {
    quarter: "Q2 · 2026",
    title: "Nebula-2 — long-horizon tasks",
    status: "active",
    body: "Doubling sequence length to 8 minutes, adding 60 new long-horizon assembly and household tasks across 12 sites.",
    artefacts: ["task taxonomy v2", "site capture rigs", "alignment toolkit"],
  },
  {
    quarter: "Q3 · 2026",
    title: "Cross-embodiment evaluation suite",
    status: "queued",
    body: "Standardised benchmarks across 37 robot embodiments, with reproducible sim-to-real transfer scoring.",
    artefacts: ["eval harness", "leaderboard", "transfer report"],
  },
  {
    quarter: "Q4 · 2026",
    title: "Federated annotation network",
    status: "queued",
    body: "Community-driven annotation contributions with cryptographic provenance and contributor attribution.",
    artefacts: ["contrib protocol", "provenance ledger"],
  },
];
