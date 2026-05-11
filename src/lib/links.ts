// Source-of-truth for the project's external destinations. Imported by
// Navbar / Footer / Members / DataGallery so we never hard-code an URL
// twice.
//
// These are the *defaults* baked into the bundle. At runtime the
// `useConfig()` hook (src/lib/useConfig.ts) replaces them with values
// fetched from `GET /api/config`, so the admin can flip them without
// redeploying. The defaults below act as a fallback when the API is
// unreachable (e.g. during initial page paint, offline preview, or
// static export).
//
// Preview-release policy (HumanNet): expose the project GitHub account
// and waitlist. The arXiv paper is also live; admin tabs still let an
// operator fill the remaining channels in later without code changes.

export const DEFAULT_LINKS = {
  github: "https://github.com/DAGroup-PKU",
  huggingface: "#",
  waitlist: "https://tally.so/r/ZjPQ4y",
  discord: "#",
  mailingList: "#",
  codeOfConduct: "#",
  arxiv: "https://arxiv.org/abs/2605.06747",
} as const;

export type SiteLinks = typeof DEFAULT_LINKS;
