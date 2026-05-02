export interface Member {
  id: string;
  name: string;
  initials: string;
  role: string;
  org: string;
  focus: string;
}

// Per-person team listing. The public Members section currently
// renders two organisations (PKU DAGroup + SimpleSilicon) hard-coded
// in src/components/Members.tsx, so this array is unused on the
// visitor side for the Preview release. We keep one placeholder
// entry so the admin Team tab is non-empty out of the box.
export const MEMBERS: Member[] = [
  {
    id: "humannet-core",
    name: "HumanNet Core Team",
    initials: "HN",
    role: "Maintainers",
    org: "PKU DAGroup · SimpleSilicon",
    focus: "Per-person credits land alongside the methods paper.",
  },
];
