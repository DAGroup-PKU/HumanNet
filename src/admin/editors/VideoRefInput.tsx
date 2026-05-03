import type { StoredVideo } from "../../lib/config-types";
import { Field, NativeSelect, TextInput } from "./Field";

interface Props {
  value: StoredVideo;
  onChange: (next: StoredVideo) => void;
  /** Title shown above the source-type selector. */
  label?: string;
}

type Kind = StoredVideo["kind"];

const KIND_OPTIONS: { id: Kind; label: string }[] = [
  { id: "local", label: "Local · bundled with site (/videos/…)" },
  { id: "external", label: "External · public URL (CDN, public OSS, etc.)" },
  { id: "oss", label: "OSS · private bucket, signed at request time" },
];

const KIND_HINT: Record<Kind, string> = {
  local:
    "Path under /public, served by the site itself. Use only for sample clips bundled with the build.",
  external:
    "Any absolute URL the browser can fetch directly. No signing — the URL is public forever.",
  oss:
    "Aliyun OSS object reference. The server serves it through /api/clip/<id>.mp4, so the bucket/key stay private. Bucket may be omitted to use the OSS_BUCKET env default.",
};

export function VideoRefInput({ value, onChange, label = "Video source" }: Props) {
  function setKind(kind: Kind) {
    if (kind === value.kind) return;
    if (kind === "local") onChange({ kind: "local", path: "/videos/" });
    else if (kind === "external")
      onChange({ kind: "external", url: "https://" });
    else onChange({ kind: "oss", key: "" });
  }

  return (
    <div className="grid gap-3 rounded-md border border-nebula-line bg-nebula-bg p-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
          {label}
        </span>
        <span
          className={`rounded-sm border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.22em] ${
            value.kind === "oss"
              ? "border-nebula-success/40 bg-nebula-success/10 text-nebula-success"
              : value.kind === "external"
                ? "border-nebula-accent/40 bg-nebula-accent/10 text-nebula-accent"
                : "border-nebula-line bg-nebula-surface text-nebula-on-muted"
          }`}
        >
          {value.kind}
        </span>
      </div>

      <Field label="Source type">
        <NativeSelect<Kind>
          value={value.kind}
          options={KIND_OPTIONS}
          onChange={(v) => setKind(v)}
        />
      </Field>

      {value.kind === "local" && (
        <Field label="Local path" hint="Under /public, e.g. /videos/exo/foo.mp4">
          <TextInput
            value={value.path}
            onChange={(v) => onChange({ kind: "local", path: v })}
            placeholder="/videos/exo/foo.mp4"
          />
        </Field>
      )}

      {value.kind === "external" && (
        <Field label="Absolute URL" hint="https://… (CDN, public OSS, data: URL)">
          <TextInput
            value={value.url}
            onChange={(v) => onChange({ kind: "external", url: v })}
            placeholder="https://cdn.example.com/clips/foo.mp4"
          />
        </Field>
      )}

      {value.kind === "oss" && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Field
            label="Bucket"
            hint="Optional · falls back to OSS_BUCKET env on the server."
          >
            <TextInput
              value={value.bucket ?? ""}
              onChange={(v) =>
                onChange({
                  kind: "oss",
                  bucket: v.trim() === "" ? undefined : v.trim(),
                  key: value.key,
                })
              }
              placeholder="nebula-clips"
            />
          </Field>
          <Field
            label="Object key"
            hint="No leading slash. Example: hero/main-2026-q3.mp4"
          >
            <TextInput
              value={value.key}
              onChange={(v) =>
                onChange({ kind: "oss", bucket: value.bucket, key: v })
              }
              placeholder="hero/main.mp4"
            />
          </Field>
        </div>
      )}

      <p className="font-mono text-[10px] leading-relaxed uppercase tracking-[0.18em] text-nebula-on-dim">
        → {KIND_HINT[value.kind]}
      </p>
    </div>
  );
}
