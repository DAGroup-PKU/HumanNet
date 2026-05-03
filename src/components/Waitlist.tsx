import { Card } from "@heroui/react";
import { Section } from "./Section";
import { LinkButton } from "./LinkButton";
import { useConfig } from "../lib/useConfig";

const BENEFITS = [
  "Early access notifications as new HumanNet shards become available.",
  "Monthly summary of curated subsets and benchmark updates.",
  "Direct line into the maintainers' triage channel (best-effort).",
  "No ads. No partner offers. We don't share or sell your address.",
];

export function Waitlist() {
  const { links } = useConfig();
  const formUrl = links.waitlist;
  const formHost = (() => {
    try {
      return new URL(formUrl).host;
    } catch {
      return formUrl;
    }
  })();
  return (
    <Section
      id="waitlist"
      eyebrow="Access · waitlist"
      title={
        <>
          Get the next dataset drop{" "}
          <span className="text-nebula-primary">before the public mirror.</span>
        </>
      }
      description="Subscribers receive updates when new corpora, curated subsets, and capture-rig changelogs are ready. Unsubscribe anytime."
    >
      <div className="grid gap-10 lg:grid-cols-12 lg:gap-12">
        <div className="lg:col-span-5">
          <ul className="space-y-5 text-sm leading-relaxed text-nebula-on-muted">
            {BENEFITS.map((b) => (
              <li key={b} className="flex items-start gap-3">
                <span
                  aria-hidden="true"
                  className="mt-2 block h-1.5 w-1.5 shrink-0 rounded-full bg-nebula-primary"
                />
                <span>{b}</span>
              </li>
            ))}
          </ul>
        </div>

        <Card
          variant="default"
          className="bg-nebula-surface ring-1 ring-inset ring-nebula-line lg:col-span-7"
        >
          <Card.Header>
            <Card.Title className="font-display text-2xl text-nebula-on">
              Request early access
            </Card.Title>
            <Card.Description className="text-sm text-nebula-on-muted">
              We usually respond within a few business days.
            </Card.Description>
          </Card.Header>

          <Card.Content>
            <div className="flex flex-col gap-6">
              <p className="text-sm leading-relaxed text-nebula-on-muted">
                The waitlist is hosted on{" "}
                <span className="font-mono text-nebula-on">Tally</span> so
                signing up takes about 30 seconds and doesn&apos;t require an
                account. We collect email + affiliation + role, and reply from
                a real human inbox — never an autoresponder.
              </p>

              <dl className="grid grid-cols-1 gap-3 font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim sm:grid-cols-3">
                <div className="rounded-sm border border-nebula-line bg-nebula-bg px-3 py-2.5">
                  <dt>Form host</dt>
                  <dd className="mt-1 text-nebula-on">Tally</dd>
                </div>
                <div className="rounded-sm border border-nebula-line bg-nebula-bg px-3 py-2.5">
                  <dt>Time</dt>
                  <dd className="mt-1 text-nebula-on">~30 sec</dd>
                </div>
                <div className="rounded-sm border border-nebula-line bg-nebula-bg px-3 py-2.5">
                  <dt>Account</dt>
                  <dd className="mt-1 text-nebula-on">Not required</dd>
                </div>
              </dl>
            </div>
          </Card.Content>

          <Card.Footer className="mt-2 flex flex-col items-stretch gap-3 sm:mt-4">
            <LinkButton
              href={formUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full"
            >
              Open the waitlist form →
            </LinkButton>
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
              → Opens {formHost} in a new tab
            </p>
          </Card.Footer>
        </Card>
      </div>
    </Section>
  );
}
