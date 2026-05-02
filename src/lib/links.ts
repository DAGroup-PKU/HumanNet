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
// Preview-release policy (HumanNet): the only two destinations exposed
// to visitors are `waitlist` (Tally-hosted form) and `arxiv` (the
// methods paper). Everything else stays as a "#" placeholder until the
// real channel exists; admin tabs still let an operator fill them in
// later without code changes.

export const DEFAULT_LINKS = {
  github: "#",
  huggingface: "#",
  waitlist: "https://tally.so/r/humannet-waitlist",
  discord: "#",
  mailingList: "#",
  codeOfConduct: "#",
  arxiv: "#",
} as const;

export type SiteLinks = typeof DEFAULT_LINKS;
