const COL_PROJECT = [
  { label: "Dataset card", href: "#" },
  { label: "Loader SDK", href: "#" },
  { label: "Calibration tool", href: "#" },
  { label: "Benchmarks", href: "#" },
];

const COL_COMMUNITY = [
  { label: "GitHub", href: "https://github.com" },
  { label: "Discord", href: "#" },
  { label: "Mailing list", href: "#" },
  { label: "Code of conduct", href: "#" },
];

const COL_RESEARCH = [
  { label: "Methods paper", href: "#" },
  { label: "Capture rigs", href: "#" },
  { label: "Annotation schema", href: "#" },
  { label: "Reproducibility", href: "#" },
];

export function Footer() {
  return (
    <footer
      role="contentinfo"
      className="relative border-t border-nebula-line bg-nebula-base"
    >
      <div className="mx-auto w-full max-w-7xl px-6 py-16 sm:px-10 sm:py-20 lg:px-16">
        <div className="grid gap-12 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <div className="flex items-center gap-3">
              <span
                aria-hidden="true"
                className="grid h-8 w-8 place-items-center rounded-sm bg-nebula-primary/10 ring-1 ring-nebula-primary/40"
              >
                <span className="block h-2.5 w-2.5 rounded-full bg-nebula-primary shadow-[0_0_12px_rgba(238,159,50,0.55)]" />
              </span>
              <span className="font-display text-lg text-nebula-on">
                Project <span className="text-nebula-primary">Nebula</span>
              </span>
            </div>
            <p className="mt-5 max-w-md text-sm leading-relaxed text-nebula-on-muted">
              An open-source embodied AI dataset & toolchain. Released under
              CC-BY-SA 4.0 (data) and Apache 2.0 (code). Built by an open
              community of robotics researchers, engineers, and field
              operators.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <span className="rounded-sm border border-nebula-line bg-nebula-surface px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                CC-BY-SA 4.0 · data
              </span>
              <span className="rounded-sm border border-nebula-line bg-nebula-surface px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                Apache 2.0 · code
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 lg:col-span-8">
            <FooterColumn title="Project" items={COL_PROJECT} />
            <FooterColumn title="Community" items={COL_COMMUNITY} />
            <FooterColumn title="Research" items={COL_RESEARCH} />
          </div>
        </div>

        <div className="mt-16 flex flex-col items-start gap-4 border-t border-nebula-line pt-8 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim sm:flex-row sm:items-center sm:justify-between">
          <span>© 2025–2026 Project Nebula contributors.</span>
          <span>v0.9 · preview · embargo 2026-Q3</span>
        </div>
      </div>
    </footer>
  );
}

function FooterColumn({
  title,
  items,
}: {
  title: string;
  items: { label: string; href: string }[];
}) {
  return (
    <div>
      <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
        {title}
      </div>
      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item.label}>
            <a
              href={item.href}
              className="text-sm text-nebula-on-muted transition-colors hover:text-nebula-on"
            >
              {item.label}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
