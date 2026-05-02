import { useEffect, useRef, useState } from "react";

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
  /**
   * When true, defer rendering the underlying `<source>` until the tile
   * is about to enter the viewport. This avoids 40+ simultaneous OSS
   * `preload="metadata"` requests on the gallery page; with lazy=true,
   * each request only fires once the user scrolls within ~400px of the
   * tile. Once a tile has been activated it stays loaded — leaving the
   * viewport pauses but does not unload, so scrolling back is instant.
   *
   * Default false so non-gallery callers (Hero, PerspectiveExplorer)
   * keep their above-the-fold behaviour.
   */
  lazy?: boolean;
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
  lazy = false,
}: VideoFrameProps) {
  const ref = useRef<HTMLVideoElement | null>(null);
  // Sticky "this tile has been near the viewport at least once" flag.
  // When lazy=false we initialise to true so the <source> renders
  // immediately (Hero / PerspectiveExplorer behaviour). When lazy=true
  // we wait for the first IntersectionObserver hit to flip it.
  const [activated, setActivated] = useState(!lazy);

  // Lazy-activation observer — fires once with a generous rootMargin so
  // the metadata request races the user's scroll instead of waiting for
  // the tile to fully enter the viewport. Disconnects after the first
  // hit so we don't keep observing a tile that is already loaded.
  useEffect(() => {
    if (!lazy || activated) return;
    const v = ref.current;
    if (!v) return;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActivated(true);
            obs.disconnect();
            return;
          }
        }
      },
      { rootMargin: "400px 0px 400px 0px", threshold: 0 },
    );
    obs.observe(v);
    return () => obs.disconnect();
  }, [lazy, activated]);

  // Play / pause observer — runs on every tile that has pauseOffscreen
  // enabled. With lazy=true we still attach this immediately; .play()
  // simply fails (silently) until the <source> is in the DOM, and the
  // next intersection event after activation will start playback.
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
        className={`absolute inset-0 h-full w-full object-contain select-none transition-opacity duration-300 ${
          activated ? "opacity-100" : "opacity-0"
        }`}
        autoPlay={autoPlay}
        loop
        muted
        playsInline
        // Until the lazy gate flips, preload="none" so the browser does
        // *not* even fire the metadata HEAD/Range request. After
        // activation, switch to "metadata" (the original behaviour) to
        // get duration + first frame ready quickly.
        preload={activated ? "metadata" : "none"}
        poster={poster}
        aria-label={ariaLabel}
        controlsList="nodownload noplaybackrate noremoteplayback"
        disablePictureInPicture
        disableRemotePlayback
        onContextMenu={(e) => e.preventDefault()}
        draggable={false}
      >
        {/* The <source> is what triggers the network request. Gating
            its render is what actually defers the load — preload="none"
            alone is not enough on Safari. */}
        {activated ? <source src={src} type="video/mp4" /> : null}
      </video>

      {/* Dormant-state placeholder. Only renders while the lazy gate
          hasn't flipped yet (i.e. the tile is still well below the
          viewport). Without this, the dormant tile would just be a flat
          dark rectangle — readable but visually dead.
          Goals:
          - communicate "this is going to load, scroll a bit further"
          - keep the chrome on-brand (mono eyebrow, dotted grid, pulsing
            indicator) without animating expensive properties
          - aria-hidden because the <video> below carries the real label */}
      {!activated && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 bg-grid-dots"
        >
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-nebula-primary/40" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-nebula-primary/70" />
          </span>
          <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
            queued · scroll to play
          </span>
        </div>
      )}

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
