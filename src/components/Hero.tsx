import { Chip } from "@heroui/react";
import { LinkButton } from "./LinkButton";
import { VideoFrame } from "./VideoFrame";

export function Hero() {
  return (
    <section
      id="top"
      aria-labelledby="hero-title"
      className="relative scroll-mt-24 overflow-hidden border-b border-nebula-line bg-nebula-base pt-24 lg:pt-28"
    >
      {/* Soft scientific grid backdrop */}
      <div
        aria-hidden="true"
        className="bg-grid-dots pointer-events-none absolute inset-0 opacity-50"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 h-64 bg-gradient-to-b from-nebula-primary/10 via-transparent to-transparent"
      />

      <div className="relative mx-auto grid w-full max-w-7xl gap-12 px-6 py-16 sm:px-10 sm:py-20 lg:grid-cols-12 lg:gap-10 lg:px-16 lg:py-28">
        <div className="lg:col-span-7">
          <Chip
            size="sm"
            variant="tertiary"
            className="mb-6 border border-nebula-line bg-nebula-surface text-nebula-on-muted"
          >
            <span className="font-mono text-[10px] tracking-[0.22em] uppercase">
              Open Source · Embodied AI Dataset · v0.9 preview
            </span>
          </Chip>

          <h1
            id="hero-title"
            className="text-balance font-display text-[clamp(2.4rem,5.6vw,4.5rem)] font-medium leading-[1.02] tracking-[-0.025em] text-nebula-on"
          >
            The data substrate for{" "}
            <span className="text-nebula-primary">embodied intelligence</span>{" "}
            that survives contact with the real world.
          </h1>

          <p className="mt-6 max-w-2xl text-pretty text-base leading-relaxed text-nebula-on-muted sm:text-lg">
            Project Nebula publishes a fully open, multi-modal capture of how
            humans and robots actually move, manipulate, and assemble — with
            synchronised exocentric and egocentric streams, raw enough to train
            on, structured enough to evaluate against.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-3">
            <LinkButton href="#waitlist">Request early access</LinkButton>
            <LinkButton href="#perspectives" variant="tertiary">
              Inspect the data
            </LinkButton>
          </div>

          <dl className="mt-12 grid grid-cols-2 gap-x-10 gap-y-6 sm:max-w-xl sm:grid-cols-3">
            {[
              { k: "Hours", v: "12.8k" },
              { k: "Tasks", v: "230" },
              { k: "Robots", v: "37" },
            ].map((m) => (
              <div key={m.k}>
                <dt className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
                  {m.k}
                </dt>
                <dd className="mt-1 font-display text-2xl text-nebula-on tabular-nums sm:text-3xl">
                  {m.v}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="lg:col-span-5">
          <div className="relative">
            <span
              aria-hidden="true"
              className="absolute -left-3 top-0 hidden h-full w-px bg-gradient-to-b from-transparent via-nebula-line-strong to-transparent lg:block"
            />
            <div className="mb-3 flex items-center gap-2">
              <span className="block h-1.5 w-1.5 rounded-full bg-nebula-primary shadow-[0_0_12px_rgba(238,159,50,0.6)]" />
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                live · raw exocentric capture · 30 fps
              </span>
            </div>
            <VideoFrame
              src="/videos/exo/QSuxYRr3n7o_85.mp4"
              aspectRatio={2134 / 1280}
              ariaLabel="Exocentric raw capture preview"
              className="shadow-[0_30px_80px_-30px_rgba(238,159,50,0.35)]"
            />
            <div className="mt-3 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
              <span>clip · QSuxYRr3n7o_85</span>
              <span>nebula-1 / exo</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
