import { useState } from "react";
import { Tabs } from "@heroui/react";
import { Section } from "./Section";
import { VideoFrame } from "./VideoFrame";
import { PERSPECTIVES, type PerspectiveId } from "../data/perspectives";

export function PerspectiveExplorer() {
  const [active, setActive] = useState<PerspectiveId>("exo");
  const current = PERSPECTIVES.find((p) => p.id === active)!;

  return (
    <Section
      id="perspectives"
      eyebrow="Data · perspectives"
      title={
        <>
          Two viewpoints,{" "}
          <span className="text-nebula-primary">one synchronised reality.</span>
        </>
      }
      description="Embodied learning needs both the actor's view and the observer's view. Toggle between exocentric studio captures and egocentric six-panel research clips — the full frame is preserved in either case."
    >
      <Tabs
        defaultSelectedKey="exo"
        onSelectionChange={(k) => setActive(k as PerspectiveId)}
        className="w-full"
      >
        <Tabs.ListContainer>
          <Tabs.List
            aria-label="Choose a perspective"
            className="w-fit gap-1 rounded-md border border-nebula-line bg-nebula-surface p-1"
          >
            {PERSPECTIVES.map((p) => (
              <Tabs.Tab
                key={p.id}
                id={p.id}
                className="rounded-sm px-4 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-nebula-on-muted data-[selected=true]:bg-nebula-primary/15 data-[selected=true]:text-nebula-on"
              >
                {p.label} · {p.short}
                <Tabs.Indicator />
              </Tabs.Tab>
            ))}
          </Tabs.List>
        </Tabs.ListContainer>

        {PERSPECTIVES.map((p) => (
          <Tabs.Panel key={p.id} id={p.id} className="pt-8">
            <div className="grid gap-8 lg:grid-cols-12 lg:gap-10">
              <div className="lg:col-span-7">
                <div className="mb-3 flex items-center gap-3">
                  <span className="block h-1.5 w-1.5 rounded-full bg-nebula-primary" />
                  <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                    {p.label.toLowerCase()} · primary feed · raw
                  </span>
                </div>
                <VideoFrame
                  src={p.primaryVideo}
                  aspectRatio={p.aspectRatio}
                  ariaLabel={`${p.label} primary capture preview`}
                />
                <p className="mt-4 text-sm leading-relaxed text-nebula-on-muted">
                  {p.description}
                </p>
              </div>

              <div className="lg:col-span-5">
                <h3 className="font-display text-2xl text-nebula-on">
                  {p.label}
                  <span className="ml-2 font-mono text-xs uppercase tracking-[0.18em] text-nebula-on-dim">
                    / {p.short.toLowerCase()}
                  </span>
                </h3>
                <p className="mt-3 text-base leading-relaxed text-nebula-on-muted">
                  {p.tagline}
                </p>

                <dl className="mt-6 divide-y divide-nebula-line border-y border-nebula-line">
                  {p.bullets.map((b) => (
                    <div
                      key={b.label}
                      className="flex items-start gap-6 py-4"
                    >
                      <dt className="w-32 shrink-0 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
                        {b.label}
                      </dt>
                      <dd className="text-sm text-nebula-on">{b.value}</dd>
                    </div>
                  ))}
                </dl>

                <div className="mt-6 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
                  <span>active · {active}</span>
                  <span>humannet · {p.id}</span>
                </div>
              </div>
            </div>
          </Tabs.Panel>
        ))}
      </Tabs>

      {/* fallback for screen-readers / reduce motion users */}
      <div className="sr-only" aria-live="polite">
        Currently showing {current.label} ({current.short}) data.
      </div>
    </Section>
  );
}
