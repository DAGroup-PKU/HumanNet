import type { Member } from "../../data/members";
import { Field, TextArea, TextInput } from "./Field";

interface Props {
  value: Member[];
  onChange: (next: Member[]) => void;
}

function blankMember(): Member {
  return {
    id: `m${Date.now().toString(36)}`,
    name: "New Contributor",
    initials: "NC",
    role: "Role",
    org: "Organisation",
    focus: "What they work on.",
  };
}

export function TeamEditor({ value, onChange }: Props) {
  function patch(idx: number, patch: Partial<Member>) {
    onChange(value.map((m, i) => (i === idx ? { ...m, ...patch } : m)));
  }
  function add() {
    onChange([...value, blankMember()]);
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
          Order shown here is the order rendered on the public site. Drag-free
          reorder via ↑ / ↓ buttons.
        </p>
        <button
          type="button"
          onClick={add}
          className="ml-auto rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
        >
          + Add member
        </button>
      </div>

      <div className="grid gap-4">
        {value.map((m, i) => (
          <article
            key={m.id + i}
            className="rounded-md border border-nebula-line bg-nebula-surface p-5"
          >
            <header className="mb-4 flex flex-wrap items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                #{i + 1}
              </span>
              <span className="font-display text-base text-nebula-on">
                {m.name || "(unnamed)"}
              </span>
              <div className="ml-auto flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => move(i, -1)}
                  disabled={i === 0}
                  className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-1 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                  aria-label="Move up"
                >
                  ↑
                </button>
                <button
                  type="button"
                  onClick={() => move(i, 1)}
                  disabled={i === value.length - 1}
                  className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-1 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                  aria-label="Move down"
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
              <Field label="ID" hint="Stable, alphanumeric (no spaces).">
                <TextInput value={m.id} onChange={(v) => patch(i, { id: v })} />
              </Field>
              <Field label="Name" className="md:col-span-2">
                <TextInput
                  value={m.name}
                  onChange={(v) => patch(i, { name: v })}
                />
              </Field>
              <Field label="Initials" hint="2 chars, shown in the avatar tile.">
                <TextInput
                  value={m.initials}
                  onChange={(v) => patch(i, { initials: v })}
                />
              </Field>
              <Field label="Role">
                <TextInput
                  value={m.role}
                  onChange={(v) => patch(i, { role: v })}
                />
              </Field>
              <Field label="Organisation">
                <TextInput value={m.org} onChange={(v) => patch(i, { org: v })} />
              </Field>
              <Field label="Focus" className="md:col-span-3">
                <TextArea
                  value={m.focus}
                  onChange={(v) => patch(i, { focus: v })}
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
