import { Chip } from "@heroui/react";
import { AnimatedNumber } from "./AnimatedNumber";
import { LinkButton } from "./LinkButton";
import { VideoFrame } from "./VideoFrame";
import { useConfig } from "../lib/useConfig";
import { ACCENT_CLOSE, ACCENT_OPEN } from "../lib/config-types";

/**
 * Splits the config-driven hero title into segments so the fragment
 * wrapped in ⟦…⟧ is rendered in the brand orange. Anything between the
 * markers becomes the accented span; everything else is plain text.
 *
 * Example title:
 *   "The data substrate for ⟦embodied intelligence⟧ that survives …"
 * → ["The data substrate for ", { accent: "embodied intelligence" }, " that survives …"]
 */
function renderAccentTitle(raw: string) {
  const segments: Array<{ text: string; accent: boolean }> = [];
  let i = 0;
  while (i < raw.length) {
    const open = raw.indexOf(ACCENT_OPEN, i);
    if (open === -1) {
      segments.push({ text: raw.slice(i), accent: false });
      break;
    }
    if (open > i) segments.push({ text: raw.slice(i, open), accent: false });
    const close = raw.indexOf(ACCENT_CLOSE, open + 1);
    if (close === -1) {
      segments.push({ text: raw.slice(open + 1), accent: true });
      break;
    }
    segments.push({ text: raw.slice(open + 1, close), accent: true });
    i = close + 1;
  }
  return segments;
}

export function Hero() {
  const { hero } = useConfig();
  const titleSegments = renderAccentTitle(hero.title);
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
              {hero.eyebrow}
            </span>
          </Chip>

          <h1
            id="hero-title"
            className="text-balance font-display text-[clamp(2.4rem,5.6vw,4.5rem)] font-medium leading-[1.02] tracking-[-0.025em] text-nebula-on"
          >
            {titleSegments.map((seg, idx) =>
              seg.accent ? (
                <span key={idx} className="text-nebula-primary">
                  {seg.text}
                </span>
              ) : (
                <span key={idx}>{seg.text}</span>
              ),
            )}
          </h1>

          <p className="mt-6 max-w-2xl text-pretty text-base leading-relaxed text-nebula-on-muted sm:text-lg">
            {hero.description}
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-3">
            <LinkButton href="#waitlist">Request early access</LinkButton>
            <LinkButton href="#perspectives" variant="tertiary">
              Inspect the data
            </LinkButton>
          </div>

          <dl className="mt-12 grid grid-cols-2 gap-x-10 gap-y-6 sm:max-w-xl sm:grid-cols-3">
            {hero.metrics.map((m) => (
              <div key={m.key}>
                <dt className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
                  {m.key}
                </dt>
                <dd className="mt-1 font-display text-2xl text-nebula-on tabular-nums sm:text-3xl">
                  <AnimatedNumber value={m.value} />
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
              src={hero.primaryVideo}
              aspectRatio={hero.primaryVideoAspect}
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
