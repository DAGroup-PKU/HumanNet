export interface Member {
  id: string;
  name: string;
  initials: string;
  role: string;
  org: string;
  focus: string;
}

export const MEMBERS: Member[] = [
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
];
