import { useMemo, useState } from "react";
import { Section } from "./Section";
import { VideoFrame } from "./VideoFrame";
import { useConfig } from "../lib/useConfig";
import type { PerspectiveId } from "../data/perspectives";

type Filter = "all" | PerspectiveId;

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All clips" },
  { id: "exo", label: "Exocentric" },
  { id: "ego", label: "Egocentric" },
];

export function DataGallery() {
  const { gallery } = useConfig();
  const [filter, setFilter] = useState<Filter>("all");

  const clips = useMemo(
    () =>
      filter === "all" ? gallery : gallery.filter((c) => c.perspective === filter),
    [filter, gallery],
  );

  return (
    <Section
      id="gallery"
      eyebrow="Data · gallery"
      title={
        <>
          Sampled directly from the public bucket.{" "}
          <span className="text-nebula-on-muted">No demo reels.</span>
        </>
      }
      description="A scrollable selection of raw clips from Nebula-1 — exactly what researchers receive after running `nebula download`. Filter by perspective; every tile preserves the source aspect ratio."
    >
      <div className="mb-8 flex flex-wrap items-center gap-2">
        {FILTERS.map((f) => {
          const active = f.id === filter;
          return (
            <button
              key={f.id}
              type="button"
              aria-pressed={active}
              onClick={() => setFilter(f.id)}
              className={`rounded-sm border px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                active
                  ? "border-nebula-primary/60 bg-nebula-primary/15 text-nebula-on"
                  : "border-nebula-line bg-nebula-surface text-nebula-on-muted hover:border-nebula-line-strong hover:text-nebula-on"
              }`}
            >
              {f.label}
            </button>
          );
        })}
        <span className="ml-auto font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
          {clips.length} clip{clips.length === 1 ? "" : "s"}
        </span>
      </div>

      <ul
        className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        role="list"
      >
        {clips.map((clip) => (
          <li
            key={clip.id}
            className="group flex flex-col gap-3 rounded-lg border border-nebula-line bg-nebula-surface p-3 transition-colors hover:border-nebula-line-strong"
          >
            <VideoFrame
              src={clip.src}
              aspectRatio={clip.aspectRatio}
              ariaLabel={`${clip.task} clip — ${clip.caption}`}
              className="rounded-md"
            />
            <div className="flex items-start justify-between gap-3 px-1">
              <div className="min-w-0">
                <div className="truncate font-display text-sm text-nebula-on">
                  {clip.task}
                </div>
                <p className="line-clamp-2 text-xs leading-relaxed text-nebula-on-muted">
                  {clip.caption}
                </p>
              </div>
              <span className="shrink-0 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
                {clip.perspective}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </Section>
  );
}
