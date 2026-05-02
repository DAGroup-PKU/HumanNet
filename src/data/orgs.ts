// Organisations behind HumanNet. The Preview release intentionally
// surfaces orgs (not individual contributors) on the public page —
// the names below appear in the Members section and the Footer brand
// area. A per-person team listing is still available via the admin
// `team` editor and lives in the StoredSiteConfig payload, but it is
// not rendered on the public site for now.

export interface Org {
  id: string;
  /** Public-facing brand name shown on the card. */
  name: string;
  /** 2-letter monogram used inside the placeholder logo tile when no
   *  raster asset is supplied. */
  initials: string;
  /** Role of the org in the consortium — e.g. "Research Lab",
   *  "Industry Partner". */
  role: string;
  /** Short attribution shown below the role. Empty string is allowed. */
  context?: string;
  /** Optional bundled logo asset (Vite-imported URL). When omitted, the
   *  card renders a tasteful initials tile instead — same shape, same
   *  ring, no broken-image icon. */
  logoUrl?: string;
  /** Optional outbound link. When omitted, the card is non-clickable. */
  href?: string;
}
