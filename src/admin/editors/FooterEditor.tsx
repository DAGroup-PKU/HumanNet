import type {
  FooterColumn,
  FooterConfig,
  FooterLicenseTag,
  FooterLink,
  SiteLinks,
} from "../../lib/config-types";
import { Field, TextArea, TextInput } from "./Field";

interface Props {
  value: FooterConfig;
  /** Currently-saved SiteLinks values, used to preview $references and
   *  surface "unknown key" warnings without blocking the save. */
  links: SiteLinks;
  onChange: (next: FooterConfig) => void;
}

const LINK_KEYS: (keyof SiteLinks)[] = [
  "github",
  "huggingface",
  "waitlist",
  "discord",
  "mailingList",
  "codeOfConduct",
];

function previewHref(href: string, links: SiteLinks): string | null {
  if (!href.startsWith("$")) return null;
  const key = href.slice(1) as keyof SiteLinks;
  return Object.prototype.hasOwnProperty.call(links, key)
    ? links[key]
    : "(unknown $key — will render as #)";
}

export function FooterEditor({ value, links, onChange }: Props) {
  function patch<K extends keyof FooterConfig>(key: K, next: FooterConfig[K]) {
    onChange({ ...value, [key]: next });
  }

  // ─── License tags ──────────────────────────────────────────────────
  function setLicense(idx: number, label: string) {
    const licenses = value.licenses.map((l, i) =>
      i === idx ? { label } : l,
    );
    patch("licenses", licenses);
  }
  function addLicense() {
    patch("licenses", [
      ...value.licenses,
      { label: "License · scope" } as FooterLicenseTag,
    ]);
  }
  function removeLicense(idx: number) {
    patch(
      "licenses",
      value.licenses.filter((_, i) => i !== idx),
    );
  }

  // ─── Columns ───────────────────────────────────────────────────────
  function setColumn(idx: number, p: Partial<FooterColumn>) {
    const columns = value.columns.map((c, i) =>
      i === idx ? { ...c, ...p } : c,
    );
    patch("columns", columns);
  }
  function moveColumn(idx: number, dir: -1 | 1) {
    const j = idx + dir;
    if (j < 0 || j >= value.columns.length) return;
    const copy = value.columns.slice();
    [copy[idx], copy[j]] = [copy[j]!, copy[idx]!];
    patch("columns", copy);
  }
  function addColumn() {
    patch("columns", [
      ...value.columns,
      {
        id: `col-${Date.now().toString(36)}`,
        title: "New column",
        items: [],
      },
    ]);
  }
  function removeColumn(idx: number) {
    patch(
      "columns",
      value.columns.filter((_, i) => i !== idx),
    );
  }

  // ─── Items inside a column ─────────────────────────────────────────
  function setItem(colIdx: number, itemIdx: number, p: Partial<FooterLink>) {
    const items = value.columns[colIdx]!.items.map((it, i) =>
      i === itemIdx ? { ...it, ...p } : it,
    );
    setColumn(colIdx, { items });
  }
  function moveItem(colIdx: number, itemIdx: number, dir: -1 | 1) {
    const items = value.columns[colIdx]!.items.slice();
    const j = itemIdx + dir;
    if (j < 0 || j >= items.length) return;
    [items[itemIdx], items[j]] = [items[j]!, items[itemIdx]!];
    setColumn(colIdx, { items });
  }
  function addItem(colIdx: number) {
    const items = [
      ...value.columns[colIdx]!.items,
      { label: "New link", href: "#" },
    ];
    setColumn(colIdx, { items });
  }
  function removeItem(colIdx: number, itemIdx: number) {
    const items = value.columns[colIdx]!.items.filter((_, i) => i !== itemIdx);
    setColumn(colIdx, { items });
  }

  return (
    <section className="grid gap-6">
      <p className="max-w-2xl text-sm leading-relaxed text-nebula-on-muted">
        The Footer is rendered top-to-bottom: brand block + tagline + license
        chips on the left, columns on the right, copyright row at the bottom.
        Use{" "}
        <span className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-nebula-on">
          $key
        </span>{" "}
        in any link <span className="font-mono">href</span> to reference a{" "}
        <em>Site links</em> entry — keeps the same URL in sync across Navbar,
        Footer, and CTAs.
      </p>

      {/* Brand tagline */}
      <Field
        label="Brand tagline"
        hint="The paragraph under the Project Nebula mark. Plain text."
      >
        <TextArea
          value={value.brandTagline}
          onChange={(v) => patch("brandTagline", v)}
          rows={4}
        />
      </Field>

      {/* License chips */}
      <div className="rounded-md border border-nebula-line bg-nebula-surface p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
            License chips ({value.licenses.length})
          </div>
          <button
            type="button"
            onClick={addLicense}
            className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
          >
            + Add chip
          </button>
        </div>
        <div className="grid gap-2">
          {value.licenses.map((l, i) => (
            <div
              key={i}
              className="grid gap-2 rounded-sm border border-nebula-line bg-nebula-bg p-3 sm:grid-cols-[1fr_auto]"
            >
              <Field label={`Label #${i + 1}`}>
                <TextInput
                  value={l.label}
                  onChange={(v) => setLicense(i, v)}
                />
              </Field>
              <button
                type="button"
                onClick={() => removeLicense(i)}
                className="self-end rounded-sm border border-nebula-line bg-nebula-surface-2 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-dim hover:text-nebula-danger"
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Columns */}
      <div className="rounded-md border border-nebula-line bg-nebula-surface p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
            Columns ({value.columns.length})
          </div>
          <button
            type="button"
            onClick={addColumn}
            className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
          >
            + Add column
          </button>
        </div>

        <div className="grid gap-4">
          {value.columns.map((col, ci) => (
            <article
              key={col.id + ci}
              className="rounded-md border border-nebula-line bg-nebula-bg p-4"
            >
              <header className="mb-3 flex flex-wrap items-center gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                  #{ci + 1}
                </span>
                <span className="font-display text-base text-nebula-on">
                  {col.title}
                </span>
                <div className="ml-auto flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => moveColumn(ci, -1)}
                    disabled={ci === 0}
                    aria-label="Move up"
                    className="rounded-sm border border-nebula-line bg-nebula-surface px-2 py-1 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    onClick={() => moveColumn(ci, 1)}
                    disabled={ci === value.columns.length - 1}
                    aria-label="Move down"
                    className="rounded-sm border border-nebula-line bg-nebula-surface px-2 py-1 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                  >
                    ↓
                  </button>
                  <button
                    type="button"
                    onClick={() => removeColumn(ci)}
                    className="rounded-sm border border-nebula-line bg-nebula-surface px-2 py-1 font-mono text-[10px] text-nebula-on-muted hover:text-nebula-danger"
                  >
                    Remove
                  </button>
                </div>
              </header>

              <div className="grid gap-3 md:grid-cols-2">
                <Field
                  label="ID"
                  hint="Stable id (lowercase, dashes). Used as a React key."
                >
                  <TextInput
                    value={col.id}
                    onChange={(v) => setColumn(ci, { id: v })}
                  />
                </Field>
                <Field label="Title">
                  <TextInput
                    value={col.title}
                    onChange={(v) => setColumn(ci, { title: v })}
                  />
                </Field>
              </div>

              <div className="mt-4">
                <div className="mb-2 flex items-center justify-between">
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                    Items ({col.items.length})
                  </div>
                  <button
                    type="button"
                    onClick={() => addItem(ci)}
                    className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-muted hover:text-nebula-on"
                  >
                    + Add item
                  </button>
                </div>
                <div className="grid gap-2">
                  {col.items.map((it, ii) => {
                    const preview = previewHref(it.href, links);
                    return (
                      <div
                        key={ii}
                        className="grid gap-2 rounded-sm border border-nebula-line bg-nebula-surface p-3 sm:grid-cols-[1fr_2fr_auto_auto]"
                      >
                        <Field label="Label">
                          <TextInput
                            value={it.label}
                            onChange={(v) =>
                              setItem(ci, ii, { label: v })
                            }
                          />
                        </Field>
                        <Field
                          label="Href"
                          hint={
                            preview ? (
                              <>
                                → resolves to{" "}
                                <span className="text-nebula-on">
                                  {preview}
                                </span>
                              </>
                            ) : (
                              <>
                                Absolute URL, anchor (#…), or{" "}
                                <span className="text-nebula-on">$key</span>{" "}
                                where key ∈{" "}
                                <span className="text-nebula-on">
                                  {LINK_KEYS.join(", ")}
                                </span>
                              </>
                            )
                          }
                        >
                          <TextInput
                            value={it.href}
                            onChange={(v) => setItem(ci, ii, { href: v })}
                          />
                        </Field>
                        <Field label="External">
                          <label className="flex h-10 items-center gap-2 rounded-sm border border-nebula-line bg-nebula-surface px-3 text-sm text-nebula-on">
                            <input
                              type="checkbox"
                              checked={it.external ?? false}
                              onChange={(e) =>
                                setItem(ci, ii, {
                                  external: e.target.checked,
                                })
                              }
                              className="h-4 w-4 accent-nebula-primary"
                            />
                            <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-nebula-on-muted">
                              new tab
                            </span>
                          </label>
                        </Field>
                        <div className="flex items-end gap-1">
                          <button
                            type="button"
                            onClick={() => moveItem(ci, ii, -1)}
                            disabled={ii === 0}
                            aria-label="Move up"
                            className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-2 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                          >
                            ↑
                          </button>
                          <button
                            type="button"
                            onClick={() => moveItem(ci, ii, 1)}
                            disabled={ii === col.items.length - 1}
                            aria-label="Move down"
                            className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-2 font-mono text-[10px] text-nebula-on-muted disabled:opacity-30 hover:text-nebula-on"
                          >
                            ↓
                          </button>
                          <button
                            type="button"
                            onClick={() => removeItem(ci, ii)}
                            className="rounded-sm border border-nebula-line bg-nebula-surface-2 px-2 py-2 font-mono text-[10px] text-nebula-on-muted hover:text-nebula-danger"
                          >
                            ×
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>

      {/* Bottom row */}
      <div className="grid gap-4 md:grid-cols-2">
        <Field
          label="Copyright"
          hint="Bottom-row left text. Plain text — UTF-8 © works."
        >
          <TextInput
            value={value.copyright}
            onChange={(v) => patch("copyright", v)}
          />
        </Field>
        <Field
          label="Version tag"
          hint="Bottom-row right text. Use for release / status badges."
        >
          <TextInput
            value={value.versionTag}
            onChange={(v) => patch("versionTag", v)}
          />
        </Field>
      </div>
    </section>
  );
}
