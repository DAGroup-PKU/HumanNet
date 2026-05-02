import type { FooterLink } from "../lib/config-types";
import { useConfig } from "../lib/useConfig";
import logoUrl from "../assets/logo.png";

// All href references received from the API are already resolved by the
// backend — `$github` etc. become absolute URLs in renderPublicConfig.
// We still treat anything starting with "$" as defensively unresolved
// (in case the offline default bundle is in play) by falling back to "#".
function safeHref(href: string): string {
  return href.startsWith("$") ? "#" : href;
}

/** A href is "live" if it points somewhere real. We treat the bare "#"
 *  placeholder as not-yet-filled-in so we don't open a useless new tab
 *  for a destination the admin hasn't configured yet. Anchor hashes
 *  ("#intro") still count as live in-page links. */
function isLiveHref(href: string): boolean {
  if (href === "#" || href === "") return false;
  if (href.startsWith("$")) return false;
  return true;
}

export function Footer() {
  const { footer } = useConfig();
  const hasColumns = footer.columns.length > 0;
  const isSingleColumn = footer.columns.length === 1;

  return (
    <footer
      role="contentinfo"
      className="relative border-t border-nebula-line bg-nebula-base"
    >
      <div className="mx-auto w-full max-w-7xl px-6 py-16 sm:px-10 sm:py-20 lg:px-16">
        <div className="grid gap-12 lg:grid-cols-12">
          <div className={hasColumns ? "lg:col-span-5" : "lg:col-span-12"}>
            <div className="flex items-center gap-3">
              {/* PKU DAGroup brand mark, same as the navbar. */}
              <img
                src={logoUrl}
                alt="PKU DAGroup"
                className="h-10 w-10 rounded-sm bg-white object-contain p-0.5 ring-1 ring-nebula-line"
              />
              <span className="font-display text-lg text-nebula-on">
                Human<span className="text-nebula-primary">Net</span>
              </span>
            </div>
            <p className="mt-5 max-w-md text-sm leading-relaxed text-nebula-on-muted">
              {footer.brandTagline}
            </p>
            {footer.licenses.length > 0 && (
              <div className="mt-6 flex flex-wrap items-center gap-3">
                {footer.licenses.map((l, i) => (
                  <span
                    key={i}
                    className="rounded-sm border border-nebula-line bg-nebula-surface px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted"
                  >
                    {l.label}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Right side hosts the link columns. Layout adapts to count:
              · 1 column  → render as a single CTA card, right-aligned on lg+
              · 2 columns → side-by-side
              · 3+        → 3-up grid that wraps below sm */}
          {hasColumns && (
            <div
              className={
                isSingleColumn
                  ? "lg:col-span-7 lg:flex lg:justify-end"
                  : "grid grid-cols-2 gap-8 sm:grid-cols-3 lg:col-span-7"
              }
            >
              {isSingleColumn ? (
                <FooterCtaCard
                  title={footer.columns[0]!.title}
                  items={footer.columns[0]!.items}
                />
              ) : (
                footer.columns.map((col) => (
                  <FooterColumnView
                    key={col.id}
                    title={col.title}
                    items={col.items}
                  />
                ))
              )}
            </div>
          )}
        </div>

        <div className="mt-16 flex flex-col items-start gap-4 border-t border-nebula-line pt-8 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim sm:flex-row sm:items-center sm:justify-between">
          <span>{footer.copyright}</span>
          <span>{footer.versionTag}</span>
        </div>
      </div>
    </footer>
  );
}

function FooterColumnView({
  title,
  items,
}: {
  title: string;
  items: FooterLink[];
}) {
  return (
    <div>
      <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
        {title}
      </div>
      <ul className="space-y-3">
        {items.map((item, i) => {
          const live = isLiveHref(item.href);
          const externalLive = item.external && live;
          return (
            <li key={`${item.label}-${i}`}>
              <a
                href={safeHref(item.href)}
                {...(externalLive
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
                aria-disabled={!live ? "true" : undefined}
                className="text-sm text-nebula-on-muted transition-colors hover:text-nebula-on"
              >
                {item.label}
                {externalLive && (
                  <span aria-hidden="true" className="ml-1 text-nebula-on-dim">
                    ↗
                  </span>
                )}
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** Heavier visual treatment used when the footer has only ONE column.
 *  A plain text list looks lonely against the brand block on lg+, so we
 *  promote each item to a chip-style row with a hover lift and an arrow
 *  icon. The card is constrained to ~22 rem wide so it still feels
 *  contained on very wide viewports. */
function FooterCtaCard({
  title,
  items,
}: {
  title: string;
  items: FooterLink[];
}) {
  return (
    <div className="w-full lg:max-w-sm">
      <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
        {title}
      </div>
      <ul className="space-y-3">
        {items.map((item, i) => {
          const live = isLiveHref(item.href);
          const externalLive = item.external && live;
          // Placeholder links render with reduced opacity + a subtle
          // "soon" tag instead of the arrow, so the footer doesn't lie
          // about something being clickable when it isn't.
          return (
            <li key={`${item.label}-${i}`}>
              <a
                href={safeHref(item.href)}
                {...(externalLive
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
                aria-disabled={!live ? "true" : undefined}
                className={
                  live
                    ? "group flex items-center justify-between gap-3 rounded-sm border border-nebula-line bg-nebula-surface px-4 py-3 text-sm text-nebula-on-muted transition-colors hover:border-nebula-primary/40 hover:bg-nebula-surface-2 hover:text-nebula-on"
                    : "flex items-center justify-between gap-3 rounded-sm border border-dashed border-nebula-line/60 bg-nebula-surface/40 px-4 py-3 text-sm text-nebula-on-dim cursor-not-allowed"
                }
              >
                <span>{item.label}</span>
                {live ? (
                  <span
                    aria-hidden="true"
                    className="font-mono text-nebula-on-dim transition-transform group-hover:translate-x-0.5 group-hover:text-nebula-primary"
                  >
                    {item.external ? "↗" : "→"}
                  </span>
                ) : (
                  <span
                    aria-hidden="true"
                    className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim"
                  >
                    soon
                  </span>
                )}
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
