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
      description="What is in progress, what is next, and how each milestone will connect to artifacts researchers can inspect and reproduce."
    >
      {/* Timeline geometry (kept as fixed pixels so the rail and the dots
          share a single source of truth):
            - Vertical rail centre  → 12px from the <ol>'s left edge
            - Dot (12px) centre     → 12px from the <ol>'s left edge
            - Item content padding  → 36px so the rail clears all text
       */}
      <ol className="relative space-y-12 pl-9" role="list">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-[11px] top-3 bottom-3 w-px bg-gradient-to-b from-transparent via-nebula-line-strong to-transparent"
        />
        {ROADMAP.map((m) => (
          <li key={m.title} className="relative">
            <span
              aria-hidden="true"
              className={`pointer-events-none absolute -left-[30px] top-[10px] h-3 w-3 rounded-full ring-4 ${STATUS_RING[m.status]} ${STATUS_DOT[m.status]}`}
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
                  {m.artifacts.map((a) => (
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
