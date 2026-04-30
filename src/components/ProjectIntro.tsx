import { Card } from "@heroui/react";
import { Section } from "./Section";

const PILLARS = [
  {
    title: "Open by default",
    body: "Every clip, label, and calibration file ships with a permissive license. No gating, no application form, no academic-only clauses. The dataset is meant to be forked.",
    tag: "license",
  },
  {
    title: "Sim-to-real grounded",
    body: "Capture rigs are calibrated against synthetic counterparts. Each real clip is paired with a reproducible simulation seed so transfer-gap measurements are first-class data.",
    tag: "rigs",
  },
  {
    title: "Cross-embodiment",
    body: "37 platforms, from table-top arms to wheeled bases to humanoids, sharing one annotation schema. Researchers can train across embodiments without rewriting loaders.",
    tag: "platforms",
  },
  {
    title: "Long-horizon honest",
    body: "Sequence length and task structure are reported, not flattened. Multi-minute manipulations and assembly chains stay intact — no quiet trimming to make benchmarks look easy.",
    tag: "honesty",
  },
];

export function ProjectIntro() {
  return (
    <Section
      id="intro"
      eyebrow="Project · introduction"
      title={
        <>
          A research substrate, not a leaderboard.{" "}
          <span className="text-nebula-on-muted">
            Built by an open community, deployed under contact.
          </span>
        </>
      }
      description={
        <>
          Project Nebula is a multi-institution effort to record what embodied
          agents actually need: high-fidelity human and robot behaviour, paired
          across viewpoints and modalities, captured under the friction of real
          environments rather than the tidy floor plan of a demo lab.
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {PILLARS.map((p, i) => (
          <Card
            key={p.title}
            variant="default"
            className="bg-nebula-surface ring-1 ring-inset ring-nebula-line transition-colors hover:ring-nebula-line-strong"
          >
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
                {p.tag}
              </span>
            </div>
            <Card.Header className="mt-3">
              <Card.Title className="font-display text-xl text-nebula-on">
                {p.title}
              </Card.Title>
              <Card.Description className="text-sm leading-relaxed text-nebula-on-muted">
                {p.body}
              </Card.Description>
            </Card.Header>
          </Card>
        ))}
      </div>
    </Section>
  );
}
