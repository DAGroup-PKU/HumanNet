import type { SiteLinks } from "../../lib/config-types";
import { Field, TextInput } from "./Field";

interface Props {
  value: SiteLinks;
  onChange: (next: SiteLinks) => void;
}

const SPEC: { key: keyof SiteLinks; label: string; hint: string }[] = [
  {
    key: "waitlist",
    label: "Waitlist form URL",
    hint: "External form (Tally / Typeform / Notion) — opens in a new tab. Currently the primary CTA on the public site.",
  },
  {
    key: "arxiv",
    label: "arXiv paper URL",
    hint: "Full preprint URL (https://arxiv.org/abs/…). Drives the navbar 'arXiv' button. Use '#' to keep it disabled until the paper is live.",
  },
  {
    key: "github",
    label: "GitHub URL",
    hint: "Org or repo URL. Hidden from the public site for the Preview release; kept here for forward compatibility.",
  },
  {
    key: "huggingface",
    label: "Hugging Face URL",
    hint: "https://huggingface.co/<org> · placeholder for now.",
  },
  { key: "discord", label: "Discord invite", hint: "Or '#' to leave as a placeholder." },
  { key: "mailingList", label: "Mailing list URL", hint: "Or '#' to leave as a placeholder." },
  { key: "codeOfConduct", label: "Code of conduct URL", hint: "Or '#' to leave as a placeholder." },
];

export function LinksEditor({ value, onChange }: Props) {
  function patch(k: keyof SiteLinks, v: string) {
    onChange({ ...value, [k]: v });
  }

  return (
    <section className="grid gap-6">
      <p className="max-w-2xl text-sm leading-relaxed text-nebula-on-muted">
        These URLs feed every external link on the site (Navbar GitHub
        button, Members "View open issues" button, Footer columns, the
        Waitlist CTA). A change here propagates to every visitor on the
        next page load — no rebuild needed.
      </p>

      {SPEC.map((s) => (
        <Field key={s.key} label={s.label} hint={s.hint}>
          <TextInput
            value={value[s.key]}
            onChange={(v) => patch(s.key, v)}
            placeholder="https://…"
          />
        </Field>
      ))}
    </section>
  );
}
