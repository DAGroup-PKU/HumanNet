import { useEffect, useState } from "react";
import { LinkButton } from "./LinkButton";
import { useConfig } from "../lib/useConfig";
import logoUrl from "../assets/logo.png";

// Keep this short. Anything beyond ~5 starts to feel like a sitemap, not
// a navigation. Data scale + Gallery are reachable from the Perspectives
// flow; Waitlist is the right-hand CTA, so it's intentionally absent here.
const NAV_LINKS = [
  { id: "profile", label: "Project" },
  { id: "perspectives", label: "Data" },
  { id: "roadmap", label: "Roadmap" },
  { id: "members", label: "Team" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { links } = useConfig();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 4);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={[
        "fixed inset-x-0 top-0 z-50 transition-colors duration-200",
        scrolled
          ? "border-b border-nebula-line bg-nebula-base/85 backdrop-blur-md"
          : "border-b border-transparent",
      ].join(" ")}
      role="banner"
    >
      <div className="mx-auto flex h-16 w-full max-w-7xl items-center gap-4 px-6 sm:px-10 lg:h-20 lg:px-16">
        <a href="#top" className="flex items-center gap-3 font-medium text-nebula-on">
          {/* PKU DAGroup mark. The asset has a white background, so we
              render it as a small white card on the dark nav — looks
              intentional rather than alpha-cut. */}
          <img
            src={logoUrl}
            alt="PKU DAGroup"
            className="h-9 w-9 rounded-sm bg-white object-contain p-0.5 ring-1 ring-nebula-line"
          />
          <span className="font-display text-base tracking-tight">
            Human<span className="text-nebula-primary">Net</span>
          </span>
        </a>

        <nav
          aria-label="Primary"
          className="ml-auto hidden items-center gap-1 lg:flex"
        >
          {NAV_LINKS.map((link) => (
            <a
              key={link.id}
              href={`#${link.id}`}
              className="rounded-sm px-3 py-2 font-mono text-xs uppercase tracking-[0.18em] text-nebula-on-muted transition-colors hover:text-nebula-on"
            >
              {link.label}
            </a>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2 lg:ml-3">
          {/* arXiv preprint link. Falls back to a hash anchor when the
              admin hasn't filled in a real URL yet — keeps the button
              from advertising a broken destination. */}
          <LinkButton
            href={links.arxiv && links.arxiv !== "#" ? links.arxiv : "#top"}
            {...(links.arxiv && links.arxiv !== "#"
              ? { target: "_blank", rel: "noopener noreferrer" }
              : { "aria-disabled": true })}
            size="sm"
            variant="tertiary"
            className="hidden sm:inline-flex"
          >
            arXiv
          </LinkButton>
          <LinkButton href="#waitlist" size="sm">
            Join waitlist
          </LinkButton>
          <button
            type="button"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-menu"
            onClick={() => setOpen((v) => !v)}
            className="grid h-9 w-9 place-items-center rounded-sm border border-nebula-line text-nebula-on lg:hidden"
          >
            <span aria-hidden="true" className="relative block h-3 w-4">
              <span
                className={`absolute left-0 right-0 h-px origin-center bg-current transition ${
                  open ? "top-1/2 -translate-y-1/2 rotate-45" : "top-0"
                }`}
              />
              <span
                className={`absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-current transition ${
                  open ? "opacity-0" : "opacity-100"
                }`}
              />
              <span
                className={`absolute left-0 right-0 h-px origin-center bg-current transition ${
                  open ? "top-1/2 -translate-y-1/2 -rotate-45" : "bottom-0"
                }`}
              />
            </span>
          </button>
        </div>
      </div>

      <div
        id="mobile-menu"
        className={`overflow-hidden border-t border-nebula-line bg-nebula-base/95 backdrop-blur-md transition-[max-height,opacity] duration-200 ease-out lg:hidden ${
          open ? "max-h-[420px] opacity-100" : "pointer-events-none max-h-0 opacity-0"
        }`}
      >
        <ul className="flex flex-col gap-1 px-6 py-5 sm:px-10">
          {NAV_LINKS.map((link) => (
            <li key={link.id}>
              <a
                href={`#${link.id}`}
                onClick={() => setOpen(false)}
                className="block rounded-sm px-3 py-3 font-mono text-xs uppercase tracking-[0.18em] text-nebula-on-muted hover:bg-nebula-surface hover:text-nebula-on"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </header>
  );
}
