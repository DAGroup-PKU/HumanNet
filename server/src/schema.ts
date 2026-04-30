import { z } from "zod";

// Validation schema for site config writes. Kept loose enough that the
// admin can paste long descriptions, but tight enough that obvious
// rubbish (URLs as numbers, missing required fields) is rejected.

const HeroMetric = z.object({
  key: z.string().min(1).max(40),
  value: z.string().min(1).max(40),
});

const HeroConfig = z.object({
  eyebrow: z.string().min(1).max(160),
  title: z.string().min(1).max(400),
  description: z.string().min(1).max(800),
  primaryVideo: z.string().min(1).max(2048),
  primaryVideoAspect: z.number().positive().max(20),
  metrics: z.array(HeroMetric).min(1).max(8),
});

const UrlOrPlaceholder = z.string().min(1).max(2048);

const SiteLinks = z.object({
  github: UrlOrPlaceholder,
  huggingface: UrlOrPlaceholder,
  waitlist: UrlOrPlaceholder,
  discord: UrlOrPlaceholder,
  mailingList: UrlOrPlaceholder,
  codeOfConduct: UrlOrPlaceholder,
});

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

const GalleryClip = z.object({
  id: z
    .string()
    .min(1)
    .max(40)
    .regex(/^[a-z0-9_-]+$/i, "id must be alphanumeric / dash / underscore"),
  src: z.string().min(1).max(2048),
  perspective: z.enum(["exo", "ego"]),
  task: z.string().min(1).max(80),
  caption: z.string().min(1).max(280),
  aspectRatio: z.number().positive().max(20),
});

export const SiteConfigSchema = z.object({
  hero: HeroConfig,
  links: SiteLinks,
  team: z.array(Member).max(50),
  gallery: z.array(GalleryClip).max(200),
});

export type SiteConfigInput = z.infer<typeof SiteConfigSchema>;

export const LoginSchema = z.object({
  username: z.string().min(1).max(80),
  password: z.string().min(1).max(200),
});
