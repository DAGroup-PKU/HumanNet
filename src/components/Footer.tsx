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

export function Footer() {
  const { footer } = useConfig();

  return (
    <footer
      role="contentinfo"
      className="relative border-t border-nebula-line bg-nebula-base"
    >
      <div className="mx-auto w-full max-w-7xl px-6 py-16 sm:px-10 sm:py-20 lg:px-16">
        <div className="grid gap-12 lg:grid-cols-12">
          <div className="lg:col-span-4">
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

          <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 lg:col-span-8">
            {footer.columns.map((col) => (
              <FooterColumnView key={col.id} title={col.title} items={col.items} />
            ))}
          </div>
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
        {items.map((item, i) => (
          <li key={`${item.label}-${i}`}>
            <a
              href={safeHref(item.href)}
              {...(item.external
                ? { target: "_blank", rel: "noopener noreferrer" }
                : {})}
              className="text-sm text-nebula-on-muted transition-colors hover:text-nebula-on"
            >
              {item.label}
              {item.external && (
                <span aria-hidden="true" className="ml-1 text-nebula-on-dim">
                  ↗
                </span>
              )}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
