import { useEffect, useRef } from "react";
import promotionVideoUrl from "../assets/promotion-video.mp4";

export function PromotionVideo() {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          void video.play().catch(() => {});
        } else {
          video.pause();
        }
      },
      { threshold: 0.35 },
    );

    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="relative mx-auto w-full max-w-7xl px-6 pb-16 sm:px-10 sm:pb-20 lg:px-16 lg:pb-24">
      <div className="border-t border-nebula-line pt-8 sm:pt-10">
        <div className="mb-5 grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(280px,0.45fr)] lg:items-end">
          <div>
            <div className="eyebrow mb-3">Project film</div>
            <h2 className="max-w-3xl font-display text-2xl font-medium leading-tight text-nebula-on sm:text-3xl">
              HumanNet in motion, from raw human video to embodied learning data.
            </h2>
          </div>
          <p className="max-w-xl text-sm leading-relaxed text-nebula-on-muted lg:justify-self-end">
            A short overview of the dataset vision, capture perspectives, and
            release scope.
          </p>
        </div>

        <div className="group relative overflow-hidden rounded-md bg-nebula-surface-2 ring-1 ring-inset ring-nebula-line transition-colors hover:ring-nebula-line-strong">
          <video
            ref={videoRef}
            className="block aspect-video w-full bg-black object-contain"
            autoPlay
            controls
            muted
            preload="auto"
            playsInline
            controlsList="nodownload"
            src={promotionVideoUrl}
            aria-label="HumanNet project overview video"
          />
        </div>
      </div>
    </div>
  );
}
