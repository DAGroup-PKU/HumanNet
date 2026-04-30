import { Section } from "./Section";
import { ROADMAP, type RoadmapStatus } from "../data/roadmap";

const STATUS_LABEL: Record<RoadmapStatus, string> = {
  shipped: "Shipped",
  active: "In flight",
  queued: "Queued",
};

const STATUS_DOT: Record<RoadmapStatus, string> = {
  shipped: "bg-nebula-success shadow-[0_0_10px_rgba(67,196,122,0.6)]",
  active: "bg-nebula-primary shadow-[0_0_10px_rgba(238,159,50,0.7)] animate-pulse",
  queued: "bg-nebula-on-dim",
};

const STATUS_RING: Record<RoadmapStatus, string> = {
  shipped: "ring-nebula-success/40",
  active: "ring-nebula-primary/50",
  queued: "ring-nebula-line-strong",
};

export function Roadmap() {
  return (
    <Section
      id="roadmap"
      eyebrow="Project · roadmap"
      title={
        <>
          A public timeline,{" "}
          <span className="text-nebula-primary">not a press release.</span>
        </>
      }
      description="Where we shipped, where we are, and where we're heading. Each milestone is anchored to artefacts you can already (or will eventually) inspect on GitHub."
    >
      <ol className="relative space-y-10 pl-8 sm:pl-10" role="list">
        <span
          aria-hidden="true"
          className="absolute left-2 top-2 bottom-2 w-px bg-gradient-to-b from-transparent via-nebula-line-strong to-transparent sm:left-3"
        />
        {ROADMAP.map((m) => (
          <li key={m.title} className="relative">
            <span
              aria-hidden="true"
              className={`absolute -left-[3px] top-1.5 h-2.5 w-2.5 rounded-full ring-4 ${STATUS_RING[m.status]} ${STATUS_DOT[m.status]} sm:-left-[1px]`}
            />
            <div className="flex flex-col gap-3 sm:flex-row sm:items-baseline sm:gap-6">
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim sm:w-32 sm:shrink-0">
                {m.quarter}
              </div>
              <div className="flex-1">
                <div className="flex flex-wrap items-baseline gap-3">
                  <h3 className="font-display text-xl text-nebula-on sm:text-2xl">
                    {m.title}
                  </h3>
                  <span
                    className={`font-mono text-[10px] uppercase tracking-[0.22em] ${
                      m.status === "shipped"
                        ? "text-nebula-success"
                        : m.status === "active"
                          ? "text-nebula-primary"
                          : "text-nebula-on-dim"
                    }`}
                  >
                    · {STATUS_LABEL[m.status]}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-nebula-on-muted sm:text-base">
                  {m.body}
                </p>
                <ul className="mt-3 flex flex-wrap gap-2">
                  {m.artefacts.map((a) => (
                    <li
                      key={a}
                      className="rounded-sm border border-nebula-line bg-nebula-surface px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted"
                    >
                      {a}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </Section>
  );
}
