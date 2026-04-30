// Source-of-truth for the project's external destinations. Imported by
// Navbar / Footer / etc. so we never hard-code an URL twice.
//
// These are the *defaults* baked into the bundle. At runtime the
// `useConfig()` hook (src/lib/useConfig.ts) replaces them with
// values fetched from `GET /api/config`, so the admin can flip them
// without redeploying. The defaults below act as a fallback when the
// API is unreachable (e.g. during initial page paint, offline preview,
// or static export).

export const DEFAULT_LINKS = {
  github: "https://github.com/DAGroup-PKU",
  huggingface: "https://huggingface.co/DAGroup-PKU",
  waitlist: "https://tally.so/r/project-nebula-waitlist",
  // Placeholders — replace once the real channels exist.
  discord: "#",
  mailingList: "#",
  codeOfConduct: "#",
} as const;

export type SiteLinks = typeof DEFAULT_LINKS;
