import { z } from "zod";

// Validation schema for site config writes. Kept loose enough that the
// admin can paste long descriptions, but tight enough that obvious
// rubbish (URLs as numbers, missing required fields) is rejected.

// ─────────────────────────────────────────────────────────────────────
// Video reference (discriminated union)
// ─────────────────────────────────────────────────────────────────────
//
// Three variants:
//   - local:    a path under /public served by the site itself
//   - external: any direct URL (CDN / public OSS / data: URL)
//   - oss:      private OSS bucket+key the server signs at request time

const LocalRelativePath = z
  .string()
  .min(1)
  .max(2048)
  // Must start with "/" and not escape upward; admin can edit but we
  // refuse anything that smells like ../ traversal.
  .regex(/^\/[^\s]+$/u, "local path must start with / and not contain spaces")
  .refine((v) => !v.includes(".."), {
    message: "local path may not contain '..'",
  });

const AbsoluteUrl = z
  .string()
  .min(1)
  .max(2048)
  .regex(
    /^(https?:\/\/|data:|blob:)/u,
    "external url must start with http(s):// or data: / blob:",
  );

const OssBucket = z
  .string()
  .min(3)
  .max(63)
  .regex(/^[a-z0-9][a-z0-9-]*[a-z0-9]$/u, "invalid bucket name");

const OssKey = z
  .string()
  .min(1)
  .max(1024)
  // Forbid leading slash (OSS keys are not path-prefixed).
  .regex(/^[^\s/].*$/u, "key must not start with / or whitespace")
  .refine((v) => !v.includes(".."), {
    message: "key may not contain '..'",
  });

const StoredVideo = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("local"), path: LocalRelativePath }),
  z.object({ kind: z.literal("external"), url: AbsoluteUrl }),
  z.object({
    kind: z.literal("oss"),
    bucket: OssBucket.optional(),
    key: OssKey,
  }),
]);

// ─────────────────────────────────────────────────────────────────────
// Hero
// ─────────────────────────────────────────────────────────────────────

const HeroMetric = z.object({
  key: z.string().min(1).max(40),
  value: z.string().min(1).max(40),
});

const HeroConfig = z.object({
  eyebrow: z.string().min(1).max(160),
  title: z.string().min(1).max(400),
  description: z.string().min(1).max(800),
  primaryVideo: StoredVideo,
  primaryVideoAspect: z.number().positive().max(20),
  metrics: z.array(HeroMetric).min(1).max(8),
});

// ─────────────────────────────────────────────────────────────────────
// Links
// ─────────────────────────────────────────────────────────────────────

const UrlOrPlaceholder = z.string().min(1).max(2048);

const SiteLinks = z.object({
  github: UrlOrPlaceholder,
  huggingface: UrlOrPlaceholder,
  waitlist: UrlOrPlaceholder,
  discord: UrlOrPlaceholder,
  mailingList: UrlOrPlaceholder,
  codeOfConduct: UrlOrPlaceholder,
  arxiv: UrlOrPlaceholder,
});

// ─────────────────────────────────────────────────────────────────────
// Team
// ─────────────────────────────────────────────────────────────────────

const Member = z.object({
  id: z
    .string()
    .min(1)
    .max(40)
    .regex(/^[a-z0-9_-]+$/i, "id must be alphanumeric / dash / underscore"),
  name: z.string().min(1).max(80),
  initials: z.string().min(1).max(4),
  role: z.string().min(1).max(80),
  org: z.string().min(1).max(80),
  focus: z.string().min(1).max(200),
});

// ─────────────────────────────────────────────────────────────────────
// Gallery
// ─────────────────────────────────────────────────────────────────────

const GalleryClip = z.object({
  id: z
    .string()
    .min(1)
    .max(40)
    .regex(/^[a-z0-9_-]+$/i, "id must be alphanumeric / dash / underscore"),
  src: StoredVideo,
  perspective: z.enum(["exo", "ego"]),
  task: z.string().min(1).max(80),
  caption: z.string().min(1).max(280),
  aspectRatio: z.number().positive().max(20),
});

// ─────────────────────────────────────────────────────────────────────
// Footer
// ─────────────────────────────────────────────────────────────────────

// hrefs accept absolute URL, "#anchor", or "$siteLinkKey" (resolved by
// the renderer to siteConfig.links[key]). Empty hash and short labels
// like "$github" are both common, so we keep this loose.
const FooterHref = z.string().min(1).max(2048);

const FooterLink = z.object({
  label: z.string().min(1).max(80),
  href: FooterHref,
  external: z.boolean().optional(),
});

const FooterColumn = z.object({
  id: z
    .string()
    .min(1)
    .max(40)
    .regex(/^[a-z0-9_-]+$/i, "id must be alphanumeric / dash / underscore"),
  title: z.string().min(1).max(40),
  items: z.array(FooterLink).max(20),
});

const FooterLicenseTag = z.object({
  label: z.string().min(1).max(80),
});

const FooterConfig = z.object({
  brandTagline: z.string().min(1).max(800),
  licenses: z.array(FooterLicenseTag).max(8),
  // 0 columns is legal — the Preview release ships with the columns
  // hidden so visitors can focus on the brand + licence info. The
  // admin tab still allows up to 6 columns when channels are ready.
  columns: z.array(FooterColumn).max(6),
  copyright: z.string().min(1).max(160),
  versionTag: z.string().min(1).max(80),
});

// ─────────────────────────────────────────────────────────────────────
// Top-level
// ─────────────────────────────────────────────────────────────────────

export const SiteConfigSchema = z.object({
  hero: HeroConfig,
  links: SiteLinks,
  team: z.array(Member).max(50),
  gallery: z.array(GalleryClip).max(200),
  footer: FooterConfig,
});

export type SiteConfigInput = z.infer<typeof SiteConfigSchema>;

export const LoginSchema = z.object({
  username: z.string().min(1).max(80),
  password: z.string().min(1).max(200),
});
