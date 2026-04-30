import { Section } from "./Section";
import { useConfig } from "../lib/useConfig";

export function Members() {
  const { team, links } = useConfig();
  return (
    <Section
      id="members"
      eyebrow="Team · contributors"
      title={
        <>
          Built by people who run robots,{" "}
          <span className="text-nebula-on-muted">
            not just write papers about them.
          </span>
        </>
      }
      description="Project Nebula is maintained by a small core team plus a wider network of community contributors. Everyone listed here can be reached on the project Discord."
    >
      <ul
        className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
        role="list"
      >
        {team.map((m) => (
          <li
            key={m.id}
            className="group flex items-start gap-4 rounded-lg border border-nebula-line bg-nebula-surface p-5 transition-colors hover:border-nebula-line-strong sm:p-6"
          >
            <span
              aria-hidden="true"
              className="grid h-12 w-12 shrink-0 place-items-center rounded-md bg-nebula-primary/10 font-mono text-sm tracking-[0.2em] text-nebula-primary ring-1 ring-inset ring-nebula-primary/30 sm:h-14 sm:w-14 sm:text-base"
            >
              {m.initials}
            </span>
            <div className="min-w-0">
              <div className="font-display text-lg text-nebula-on">
                {m.name}
              </div>
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
                {m.role}
              </div>
              <div className="mt-1 text-xs text-nebula-on-dim">{m.org}</div>
              <p className="mt-3 text-sm leading-relaxed text-nebula-on-muted">
                {m.focus}
              </p>
            </div>
          </li>
        ))}
      </ul>

      <div className="mt-10 flex flex-col items-start gap-4 rounded-lg border border-dashed border-nebula-line bg-nebula-surface/60 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="eyebrow mb-2">Contribute</div>
          <p className="max-w-xl text-sm leading-relaxed text-nebula-on-muted">
            Researchers, students, and field operators are welcome — capture
            rigs, annotation tooling, and benchmark harnesses are all open
            workstreams.
          </p>
        </div>
        <a
          href={`${links.github}/issues`}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-sm border border-nebula-primary/40 bg-nebula-primary/10 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-nebula-primary transition-colors hover:bg-nebula-primary/20"
        >
          View open issues →
        </a>
      </div>
    </Section>
  );
}
