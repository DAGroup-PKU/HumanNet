export type RoadmapStatus = "shipped" | "active" | "queued";

export interface RoadmapMilestone {
  quarter: string;
  title: string;
  status: RoadmapStatus;
  body: string;
  artifacts: string[];
}

// HumanNet Preview roadmap. Two near-term milestones; both unshipped.
// Keeping the list intentionally short — anything we haven't started yet
// belongs in a research log, not on the public site.
export const ROADMAP: RoadmapMilestone[] = [
  {
    quarter: "Next",
    title: "Benchmark",
    status: "active",
    body: "We will build a HumanNet benchmark for reproducible evaluation across embodied perception and policy-related tasks, with baseline results and clearly defined evaluation protocols.",
    artifacts: [
      "evaluation tasks",
      "baseline results",
      "reproducible protocols",
    ],
  },
  {
    quarter: "Following",
    title: "Ego-data Scaling",
    status: "queued",
    body: "We will validate the scaling effect of HumanNet's egocentric data by training and evaluating models on controlled ego-data subsets, measuring how performance changes as data volume increases.",
    artifacts: [
      "ego-data subsets",
      "scaling analysis",
      "controlled experiments",
    ],
  },
];
