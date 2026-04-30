import type { HeroConfig } from "../../lib/config-types";
import { ACCENT_CLOSE, ACCENT_OPEN } from "../../lib/config-types";
import { Field, TextArea, TextInput } from "./Field";

interface Props {
  value: HeroConfig;
  onChange: (next: HeroConfig) => void;
}

export function HeroEditor({ value, onChange }: Props) {
  function patch<K extends keyof HeroConfig>(key: K, next: HeroConfig[K]) {
    onChange({ ...value, [key]: next });
  }

  function patchMetric(idx: number, key: "key" | "value", next: string) {
    const metrics = value.metrics.map((m, i) => (i === idx ? { ...m, [key]: next } : m));
    onChange({ ...value, metrics });
  }

  function addMetric() {
    onChange({
      ...value,
      metrics: [...value.metrics, { key: "Label", value: "0" }],
    });
  }
  function removeMetric(idx: number) {
    onChange({
      ...value,
      metrics: value.metrics.filter((_, i) => i !== idx),
    });
  }

  return (
    <section className="grid gap-6">
      <Field
        label="Eyebrow"
        hint="Small text above the headline. Stays in the chip."
      >
        <TextInput value={value.eyebrow} onChange={(v) => patch("eyebrow", v)} />
      </Field>

      <Field
        label="Title"
        hint={
          <>
            Wrap the orange-accented fragment between {ACCENT_OPEN} and{" "}
            {ACCENT_CLOSE}. Plain text otherwise.
          </>
        }
      >
        <TextArea
          value={value.title}
          onChange={(v) => patch("title", v)}
          rows={3}
        />
      </Field>

      <Field label="Description">
        <TextArea
          value={value.description}
          onChange={(v) => patch("description", v)}
          rows={4}
        />
      </Field>

      <div className="grid gap-6 md:grid-cols-3">
        <Field
          label="Primary video URL"
          hint="Absolute (OSS / CDN) or relative path under /videos."
          className="md:col-span-2"
        >
          <TextInput
            value={value.primaryVideo}
            onChange={(v) => patch("primaryVideo", v)}
            placeholder="https://… or /videos/exo/foo.mp4"
          />
        </Field>
        <Field label="Aspect ratio (w / h)" hint="e.g. 1.667 for 5:3, 1.778 for 16:9">
          <TextInput
            type="number"
            step="0.001"
            value={value.primaryVideoAspect}
            onChange={(v) =>
              patch("primaryVideoAspect", Number(v) || value.primaryVideoAspect)
            }
          />
        </Field>
      </div>

      <div className="rounded-md border border-nebula-line bg-nebula-surface p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
            Hero metrics ({value.metrics.length})
          </div>
          <button
            type="button"
            onClick={addMetric}
            className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
          >
            + Add metric
          </button>
        </div>
        <div className="grid gap-3">
          {value.metrics.map((m, i) => (
            <div
              key={i}
              className="grid gap-3 rounded-sm border border-nebula-line bg-nebula-bg p-3 sm:grid-cols-[1fr_1fr_auto]"
            >
              <Field label="Key">
                <TextInput
                  value={m.key}
                  onChange={(v) => patchMetric(i, "key", v)}
                />
              </Field>
              <Field label="Value">
                <TextInput
                  value={m.value}
                  onChange={(v) => patchMetric(i, "value", v)}
                />
              </Field>
              <button
                type="button"
                onClick={() => removeMetric(i)}
                className="self-end rounded-sm border border-nebula-line bg-nebula-surface-2 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-dim hover:text-nebula-danger"
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
