import { Section } from "./Section";
import { SCALE_METRICS } from "../data/scale";

export function DataScale() {
  return (
    <Section
      id="scale"
      eyebrow="Dataset · scale"
      title={
        <>
          Scale matters,{" "}
          <span className="text-nebula-primary">but only if it's honest.</span>
        </>
      }
      description="Headline numbers from the Nebula-1 release. Every count below maps 1-to-1 to an entry in the public dataset card — no privately held holdouts."
    >
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-nebula-line bg-nebula-line lg:grid-cols-3">
        {SCALE_METRICS.map((m, i) => (
          <div
            key={m.id}
            className="group relative bg-nebula-surface p-6 transition-colors hover:bg-nebula-surface-2 sm:p-8"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
                {String(i + 1).padStart(2, "0")} · {m.unit}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim opacity-0 transition group-hover:opacity-100">
                public
              </span>
            </div>
            <div className="mt-6 flex items-baseline gap-2">
              <span className="font-display text-4xl font-medium text-nebula-on tabular-nums sm:text-5xl">
                {m.value}
              </span>
              <span className="font-mono text-xs uppercase tracking-[0.18em] text-nebula-on-dim">
                {m.unit}
              </span>
            </div>
            <div className="mt-4 text-sm font-medium text-nebula-on">
              {m.label}
            </div>
            <p className="mt-1 text-xs leading-relaxed text-nebula-on-muted">
              {m.hint}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}
