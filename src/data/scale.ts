export interface ScaleMetric {
  id: string;
  value: string;
  unit: string;
  label: string;
  hint: string;
}

export const SCALE_METRICS: ScaleMetric[] = [
  {
    id: "hours",
    value: "12,840",
    unit: "hrs",
    label: "Annotated footage",
    hint: "across 9 task families",
  },
  {
    id: "clips",
    value: "184,520",
    unit: "clips",
    label: "Curated segments",
    hint: "balanced across exo / ego",
  },
  {
    id: "tasks",
    value: "230",
    unit: "tasks",
    label: "Embodied skills",
    hint: "manipulation · assembly · navigation",
  },
  {
    id: "robots",
    value: "37",
    unit: "platforms",
    label: "Robot embodiments",
    hint: "humanoid · mobile · table-top",
  },
  {
    id: "envs",
    value: "84",
    unit: "envs",
    label: "Real-world sites",
    hint: "labs · factories · homes · outdoor",
  },
  {
    id: "modalities",
    value: "11",
    unit: "modalities",
    label: "Sensor streams",
    hint: "RGB · depth · IMU · pose · audio · …",
  },
];
