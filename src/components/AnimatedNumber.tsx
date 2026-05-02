import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Drop-in span that count-ups a numeric string.
 *
 *   <AnimatedNumber value="12,840" />     → 0 → 12,840
 *   <AnimatedNumber value="12.8k" />      → 0.0k → 12.8k
 *   <AnimatedNumber value="184,520" />    → 0 → 184,520
 *   <AnimatedNumber value="37" />         → 0 → 37
 *   <AnimatedNumber value="11" />         → 0 → 11
 *
 * Implementation notes:
 * - The animation only starts once the element enters the viewport
 *   (IntersectionObserver). DataScale lives below the fold on mobile, so
 *   without this gating the visitor would just see the final number.
 * - Honors `prefers-reduced-motion: reduce` — no animation, render the
 *   final value immediately. Required for accessibility and a pleasant
 *   experience on low-end devices.
 * - The original formatting (commas, decimals, suffixes like "k") is
 *   preserved at every frame. We never strip these since the suffix is
 *   often what gives the metric meaning.
 * - `tabular-nums` should be applied by the *caller* on the surrounding
 *   element so the digit width doesn't twitch during the count-up.
 */
interface AnimatedNumberProps {
  value: string;
  /** Duration in ms. Default 1.4s — long enough to feel deliberate. */
  duration?: number;
  className?: string;
}

interface ParsedValue {
  numericValue: number;
  format: (n: number) => string;
}

/** Pull apart "12.8k" → { numericValue: 12.8, format: n => `${n.toFixed(1)}k` }. */
function parseValue(raw: string): ParsedValue {
  // Leading number portion: digits + optional thousands commas + optional
  // decimal portion. Suffix is whatever non-numeric text follows.
  const match = raw.match(/^([\d,]+(?:\.\d+)?)(.*)$/);
  if (!match) {
    // Fallback for strings like "—" or "N/A" — nothing to animate.
    return { numericValue: 0, format: () => raw };
  }
  const [, numStr, suffix] = match;
  const hasComma = numStr.includes(",");
  const decimalPart = numStr.split(".")[1];
  const decimalPlaces = decimalPart?.length ?? 0;
  const numericValue = parseFloat(numStr.replace(/,/g, ""));

  return {
    numericValue,
    format: (n: number) => {
      if (decimalPlaces > 0) {
        return n.toFixed(decimalPlaces) + suffix;
      }
      const rounded = Math.round(n);
      const numFormatted = hasComma
        ? rounded.toLocaleString("en-US")
        : String(rounded);
      return numFormatted + suffix;
    },
  };
}

/** easeOutCubic — feels snappier than linear without overshooting. */
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function AnimatedNumber({
  value,
  duration = 1400,
  className,
}: AnimatedNumberProps) {
  const { numericValue, format } = useMemo(() => parseValue(value), [value]);

  // Initial display: with reduced-motion (or SSR), show the final value so
  // the page never flashes "0". With motion, start at 0 and let the effect
  // animate up once the element scrolls into view.
  const initial = useMemo(() => {
    if (typeof window === "undefined") return value;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    return reduced ? value : format(0);
  }, [value, format]);

  const [display, setDisplay] = useState<string>(initial);
  const ref = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof window === "undefined") return;

    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (reduced) {
      setDisplay(value);
      return;
    }

    let raf = 0;
    let cancelled = false;

    const run = (ts0: number) => {
      const step = (ts: number) => {
        if (cancelled) return;
        const t = Math.min(1, (ts - ts0) / duration);
        setDisplay(format(numericValue * easeOut(t)));
        if (t < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };

    let started = false;
    const obs = new IntersectionObserver(
      (entries) => {
        const e = entries[0];
        if (!e?.isIntersecting || started) return;
        started = true;
        run(performance.now());
        obs.disconnect();
      },
      { threshold: 0.35 },
    );
    obs.observe(el);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      obs.disconnect();
    };
  }, [value, numericValue, format, duration]);

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  );
}
