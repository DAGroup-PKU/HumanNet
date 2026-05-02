import { Section } from "./Section";
import { useConfig } from "../lib/useConfig";
import type { Org } from "../data/orgs";
import logoUrl from "../assets/logo.png";

// HumanNet's contributor surface, Preview-edition. Two organisations
// drive the effort; per-person attribution is intentionally deferred
// until the methods paper goes live (the admin still has a `team`
// editor backing the StoredSiteConfig.team list, but that list is not
// rendered here yet).
//
// `links` is consumed via useConfig() so the GitHub URL admin sets in
// the editor flows through without a redeploy. SimpleSilicon doesn't
// expose a public org URL yet — the card stays non-clickable until one
// is ready, which keeps the layout honest instead of pretending.

export function Members() {
  const { links } = useConfig();

  const ORGS: Org[] = [
    {
      id: "pku-dagroup",
      name: "PKU DAGroup",
      initials: "DA",
      role: "Research Lab",
      context: "Peking University",
      logoUrl,
      href: links.github && links.github !== "#" ? links.github : undefined,
    },
    {
      id: "simplesilicon",
      name: "SimpleSilicon",
      initials: "SS",
      role: "Industry Partner",
      context: "Capture · curation · platform",
    },
  ];

  return (
    <Section
      id="members"
      eyebrow="Team · organisations"
      title={
        <>
          Built jointly by{" "}
          <span className="text-nebula-on-muted">
            an academic lab and an industry partner.
          </span>
        </>
      }
      description="Two organisations co-author the Preview release. Per-person credits will land alongside the methods paper."
    >
      <ul
        className="grid gap-4 sm:grid-cols-2 lg:max-w-3xl"
        role="list"
      >
        {ORGS.map((o) => (
          <OrgCard key={o.id} org={o} />
        ))}
      </ul>
    </Section>
  );
}

function OrgCard({ org }: { org: Org }) {
  // The card flips between a clickable <a> and a static <li> based on
  // whether the org has surfaced a public URL yet. Same DOM shape, just
  // different outermost tag — keeps spacing identical between the two.
  const inner = (
    <>
      {org.logoUrl ? (
        <img
          src={org.logoUrl}
          alt=""
          className="h-14 w-14 shrink-0 rounded-md bg-white object-contain p-1 ring-1 ring-nebula-line"
          draggable={false}
        />
      ) : (
        <span
          aria-hidden="true"
          className="grid h-14 w-14 shrink-0 place-items-center rounded-md bg-nebula-primary/10 font-mono text-base tracking-[0.2em] text-nebula-primary ring-1 ring-inset ring-nebula-primary/30"
        >
          {org.initials}
        </span>
      )}
      <div className="min-w-0">
        <div className="font-display text-lg text-nebula-on">{org.name}</div>
        <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
          {org.role}
        </div>
        {org.context && (
          <div className="mt-1 text-xs text-nebula-on-dim">{org.context}</div>
        )}
      </div>
      {org.href && (
        <span
          aria-hidden="true"
          className="ml-auto self-center font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim transition-transform group-hover:translate-x-0.5 group-hover:text-nebula-primary"
        >
          ↗
        </span>
      )}
    </>
  );

  if (org.href) {
    return (
      <li>
        <a
          href={org.href}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${org.name} — opens in a new tab`}
          className="group flex items-center gap-4 rounded-lg border border-nebula-line bg-nebula-surface p-5 transition-colors hover:border-nebula-line-strong sm:p-6"
        >
          {inner}
        </a>
      </li>
    );
  }
  return (
    <li className="group flex items-center gap-4 rounded-lg border border-nebula-line bg-nebula-surface p-5 sm:p-6">
      {inner}
    </li>
  );
}
