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

// First page of clips. The gallery is intentionally a *teaser* for the
// full corpus — researchers go to GitHub for the rest. Eight is enough
// to fill two rows on desktop (4-col grid) without making the page feel
// like a content dump.
const INITIAL_VISIBLE = 8;

export function DataGallery() {
  const { gallery, links } = useConfig();
  const [filter, setFilter] = useState<Filter>("all");
  const [showAll, setShowAll] = useState(false);

  const clips = useMemo(
    () =>
      filter === "all" ? gallery : gallery.filter((c) => c.perspective === filter),
    [filter, gallery],
  );

  const visible = showAll ? clips : clips.slice(0, INITIAL_VISIBLE);
  const remaining = clips.length - visible.length;
  const canCollapse = showAll && clips.length > INITIAL_VISIBLE;

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
          {visible.length} of {clips.length} clip{clips.length === 1 ? "" : "s"}
        </span>
      </div>

      <ul
        className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        role="list"
      >
        {visible.map((clip) => (
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

      {/* Reveal-or-jump-to-GitHub footer.
          - If the current filter has more than INITIAL_VISIBLE clips and we're
            collapsed, show "Load N more"; once expanded, offer a way back.
          - The right-side primary CTA always points at the GitHub corpus.
            The teaser tile is curated; the truth lives in the repo. */}
      <div className="mt-10 flex flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
        {remaining > 0 ? (
          <button
            type="button"
            onClick={() => setShowAll(true)}
            className="rounded-sm border border-nebula-line bg-nebula-surface px-4 py-2.5 text-center font-mono text-[11px] uppercase tracking-[0.22em] text-nebula-on-muted transition-colors hover:border-nebula-line-strong hover:text-nebula-on"
          >
            + Load {remaining} more clip{remaining === 1 ? "" : "s"}
          </button>
        ) : canCollapse ? (
          <button
            type="button"
            onClick={() => setShowAll(false)}
            className="rounded-sm border border-nebula-line bg-nebula-surface px-4 py-2.5 text-center font-mono text-[11px] uppercase tracking-[0.22em] text-nebula-on-muted transition-colors hover:border-nebula-line-strong hover:text-nebula-on"
          >
            Show fewer clips
          </button>
        ) : (
          <span aria-hidden="true" className="hidden sm:block" />
        )}

        <a
          href={links.github}
          target="_blank"
          rel="noopener noreferrer"
          className="group inline-flex items-center justify-center gap-2 rounded-sm border border-nebula-primary/50 bg-nebula-primary/10 px-5 py-2.5 font-mono text-[11px] uppercase tracking-[0.22em] text-nebula-on transition-colors hover:bg-nebula-primary/20"
        >
          Browse the full corpus on GitHub
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </a>
      </div>
    </Section>
  );
}
