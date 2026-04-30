import type { GalleryClip } from "../../data/gallery";
import { Field, NativeSelect, TextArea, TextInput } from "./Field";

interface Props {
  value: GalleryClip[];
  onChange: (next: GalleryClip[]) => void;
}

function blank(): GalleryClip {
  return {
    id: `clip-${Date.now().toString(36)}`,
    src: "",
    perspective: "exo",
    task: "Task",
    caption: "Short caption.",
    aspectRatio: 16 / 9,
  };
}

export function GalleryEditor({ value, onChange }: Props) {
  function patch(idx: number, p: Partial<GalleryClip>) {
    onChange(value.map((c, i) => (i === idx ? { ...c, ...p } : c)));
  }
  function add() {
    onChange([...value, blank()]);
  }
  function remove(idx: number) {
    onChange(value.filter((_, i) => i !== idx));
  }
  function move(idx: number, dir: -1 | 1) {
    const j = idx + dir;
    if (j < 0 || j >= value.length) return;
    const copy = value.slice();
    [copy[idx], copy[j]] = [copy[j]!, copy[idx]!];
    onChange(copy);
  }

  return (
    <section className="grid gap-6">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm leading-relaxed text-nebula-on-muted">
          The Gallery section renders these clips in order. Each tile preserves
          its source aspect ratio — make sure the number you enter is{" "}
          <span className="font-mono text-nebula-on">width / height</span>{" "}
          measured with <span className="font-mono">ffprobe</span>.
        </p>
        <button
          type="button"
          onClick={add}
          className="ml-auto rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
        >
          + Add clip
        </button>
      </div>

      <div className="grid gap-4">
        {value.map((c, i) => (
          <article
            key={c.id + i}
            className="rounded-md border border-nebula-line bg-nebula-surface p-5"
          >
            <header className="mb-4 flex flex-wrap items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                #{i + 1}
              </span>
              <span className="font-display text-base text-nebula-on">
                {c.task}
              </span>
              <span
                className={`rounded-sm border px-2 py-1 font-mono text-[10px] uppercase tracking-[0.22em] ${
                  c.perspective === "exo"
                    ? "border-nebula-primary/40 bg-nebula-primary/10 text-nebula-primary"
                    : "border-nebula-accent/40 bg-nebula-accent/10 text-nebula-accent"
                }`}
              >
                {c.perspective}
              </span>
              <div className="ml-auto flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => move(i, -1)}
                  disabled={i === 0}
                  aria-label="Move up"
                  className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-1 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                >
                  ↑
                </button>
                <button
                  type="button"
                  onClick={() => move(i, 1)}
                  disabled={i === value.length - 1}
                  aria-label="Move down"
                  className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-1 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                >
                  ↓
                </button>
                <button
                  type="button"
                  onClick={() => remove(i)}
                  className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-1 font-mono text-[10px] text-nebula-on-muted hover:text-nebula-danger"
                >
                  Remove
                </button>
              </div>
            </header>

            <div className="grid gap-4 md:grid-cols-3">
              <Field label="ID" hint="Unique stable identifier.">
                <TextInput value={c.id} onChange={(v) => patch(i, { id: v })} />
              </Field>
              <Field label="Task">
                <TextInput value={c.task} onChange={(v) => patch(i, { task: v })} />
              </Field>
              <Field label="Perspective">
                <NativeSelect
                  value={c.perspective}
                  onChange={(v) => patch(i, { perspective: v })}
                  options={[
                    { id: "exo", label: "Exocentric (third-person)" },
                    { id: "ego", label: "Egocentric (first-person)" },
                  ]}
                />
              </Field>
              <Field label="Source URL" className="md:col-span-2">
                <TextInput
                  value={c.src}
                  onChange={(v) => patch(i, { src: v })}
                  placeholder="https://oss-cn-…aliyuncs.com/nebula/exo/foo.mp4 or /videos/exo/foo.mp4"
                />
              </Field>
              <Field label="Aspect ratio (w / h)">
                <TextInput
                  type="number"
                  step="0.001"
                  value={c.aspectRatio}
                  onChange={(v) =>
                    patch(i, { aspectRatio: Number(v) || c.aspectRatio })
                  }
                />
              </Field>
              <Field label="Caption" className="md:col-span-3">
                <TextArea
                  value={c.caption}
                  onChange={(v) => patch(i, { caption: v })}
                  rows={2}
                />
              </Field>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
