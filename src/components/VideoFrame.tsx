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

  // Best-effort scraping deterrents. NOTE: these are UX speed bumps, not
  // security. Determined users can always pull the source URL from the
  // network tab. Real protection lives in the OSS bucket policy:
  //   - Referer whitelist on the bucket
  //   - Signed URLs with short TTL (e.g. 5-minute STS tokens)
  //   - HLS chunked playback with key rotation for paid tiers.
  // The server-side enforcement is intentionally out of this component.
  return (
    <div
      className={`relative overflow-hidden rounded-lg bg-nebula-surface-2 ring-1 ring-inset ring-nebula-line ${className}`}
      style={{ aspectRatio: `${aspectRatio}` }}
      onContextMenu={(e) => e.preventDefault()}
    >
      <video
        ref={ref}
        className="absolute inset-0 h-full w-full object-contain select-none"
        autoPlay={autoPlay}
        loop
        muted
        playsInline
        preload="metadata"
        poster={poster}
        aria-label={ariaLabel}
        controlsList="nodownload noplaybackrate noremoteplayback"
        disablePictureInPicture
        disableRemotePlayback
        onContextMenu={(e) => e.preventDefault()}
        draggable={false}
      >
        <source src={src} type="video/mp4" />
      </video>
      {/* Transparent overlay also catches right-click and drag attempts on
         non-video edges (poster, letterbox) without blocking pointer events
         on the video itself (pointer-events-none lets clicks pass through). */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 select-none"
      />
    </div>
  );
}
