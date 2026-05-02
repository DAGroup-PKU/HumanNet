export interface ScaleMetric {
  id: string;
  value: string;
  unit: string;
  label: string;
  hint: string;
}

// Headline statistics for the HumanNet Preview release. The two
// breakdown lines (TPV / FPV) sum to the total — keeping them adjacent
// in the grid lets researchers sanity-check the split at a glance
// instead of having to read a footnote.
export const SCALE_METRICS: ScaleMetric[] = [
  {
    id: "total-hours",
    value: "967k",
    unit: "hrs",
    label: "Total footage",
    hint: "third-person + egocentric, deduplicated",
  },
  {
    id: "tpv-hours",
    value: "220k",
    unit: "hrs",
    label: "Third-person view",
    hint: "Koala-36M-v1 + OpenHumanVid renders",
  },
  {
    id: "fpv-hours",
    value: "747k",
    unit: "hrs",
    label: "Egocentric view",
    hint: "helmet · GoPro · DSLR head-mount captures",
  },
  {
    id: "scenes",
    value: "30+",
    unit: "scenes",
    label: "Scene categories",
    hint: "home · industrial · work · outdoor · retail · …",
  },
  {
    id: "tasks",
    value: "720k+",
    unit: "tasks",
    label: "Task instances",
    hint: "clip-level action labels across 828 leaf categories",
  },
  {
    id: "objects",
    value: "150k",
    unit: "objects",
    label: "Object instances",
    hint: "across 800+ unique object categories",
  },
];
