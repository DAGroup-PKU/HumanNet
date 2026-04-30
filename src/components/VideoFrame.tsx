import { useEffect, useRef } from "react";

interface VideoFrameProps {
  src: string;
  /** native aspect ratio of the source clip (w/h). */
  aspectRatio: number;
  /** override container className (rounded corners, sizing) */
  className?: string;
  ariaLabel: string;
  /** Whether to autoplay. Defaults to true (muted, looped). */
  autoPlay?: boolean;
  /** Whether the video should hard-pause when it leaves the viewport. */
  pauseOffscreen?: boolean;
  poster?: string;
}

/**
 * Video player that **always shows the full frame** (no scale crop).
 *
 * Strategy:
 *   - container has `aspect-ratio` matching the source clip
 *   - the <video> uses `object-fit: contain` so the entire frame is visible
 *   - container background is the chrome's neutral surface — small letterbox
 *     bars are acceptable when the responsive width forces a different aspect
 *
 * This explicitly rejects the previous `transform: scale(3)` crop strategy
 * (per user feedback: "we need to see the entire video — slight stretching
 * is OK, but never crop content out").
 */
export function VideoFrame({
  src,
  aspectRatio,
  className = "",
  ariaLabel,
  autoPlay = true,
  pauseOffscreen = true,
  poster,
}: VideoFrameProps) {
  const ref = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const v = ref.current;
    if (!v || !pauseOffscreen) return;

    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            void v.play().catch(() => {});
          } else {
            v.pause();
          }
        }
      },
      { threshold: 0.15 },
    );
    obs.observe(v);
    return () => obs.disconnect();
  }, [pauseOffscreen]);

  return (
    <div
      className={`relative overflow-hidden rounded-lg bg-nebula-surface-2 ring-1 ring-inset ring-nebula-line ${className}`}
      style={{ aspectRatio: `${aspectRatio}` }}
    >
      <video
        ref={ref}
        className="absolute inset-0 h-full w-full object-contain"
        autoPlay={autoPlay}
        loop
        muted
        playsInline
        preload="metadata"
        poster={poster}
        aria-label={ariaLabel}
      >
        <source src={src} type="video/mp4" />
      </video>
    </div>
  );
}
