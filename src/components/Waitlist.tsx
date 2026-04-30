import { useState, type FormEvent } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  Label,
  ListBox,
  Select,
  TextField,
} from "@heroui/react";
import { Section } from "./Section";

const ROLES = [
  { id: "researcher", label: "Academic researcher" },
  { id: "industry", label: "Industry / startup" },
  { id: "student", label: "Student" },
  { id: "ops", label: "Robotics operator" },
  { id: "other", label: "Other" },
];

type Status =
  | { kind: "idle" }
  | { kind: "success"; email: string }
  | { kind: "error"; message: string };

export function Waitlist() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    const email = String(data.get("email") || "").trim();
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setStatus({ kind: "error", message: "Enter a valid email so we can reach you." });
      return;
    }
    // Simulated handshake — wire to real backend / Form provider when shipping.
    await new Promise((r) => setTimeout(r, 600));
    setStatus({ kind: "success", email });
  };

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
      description="Subscribers receive download links for new corpora 7–10 days before the public mirror sync, plus monthly capture-rig changelogs. Unsubscribe anytime."
    >
      <div className="grid gap-10 lg:grid-cols-12 lg:gap-12">
        <div className="lg:col-span-5">
          <ul className="space-y-5 text-sm leading-relaxed text-nebula-on-muted">
            {[
              "Early access to Nebula-2 long-horizon clips before public release.",
              "Monthly summary of capture rigs deployed, with reproducibility seeds.",
              "Direct line into the maintainers' triage channel (best-effort).",
              "No ads. No partner offers. We don't share or sell your address.",
            ].map((b) => (
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
              We typically respond within 3 business days.
            </Card.Description>
          </Card.Header>

          <Form onSubmit={onSubmit}>
            <Card.Content>
              <div className="flex flex-col gap-5">
                <TextField name="email" type="email" isRequired>
                  <Label className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                    Email
                  </Label>
                  <Input placeholder="you@lab.example" variant="secondary" />
                </TextField>

                <TextField name="org">
                  <Label className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                    Affiliation (optional)
                  </Label>
                  <Input
                    placeholder="University · company · independent"
                    variant="secondary"
                  />
                </TextField>

                <Select
                  name="role"
                  defaultSelectedKey="researcher"
                  placeholder="Select one"
                >
                  <Label className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
                    What best describes you?
                  </Label>
                  <Select.Trigger className="w-full">
                    <Select.Value />
                    <Select.Indicator />
                  </Select.Trigger>
                  <Select.Popover>
                    <ListBox>
                      {ROLES.map((r) => (
                        <ListBox.Item key={r.id} id={r.id} textValue={r.label}>
                          {r.label}
                          <ListBox.ItemIndicator />
                        </ListBox.Item>
                      ))}
                    </ListBox>
                  </Select.Popover>
                </Select>
              </div>
            </Card.Content>

            <Card.Footer className="mt-2 flex flex-col items-stretch gap-3 sm:mt-4">
              <Button type="submit" className="w-full font-mono text-xs uppercase tracking-[0.18em]">
                Subscribe
              </Button>
              <p
                role="status"
                aria-live="polite"
                className={`min-h-[1.25rem] font-mono text-[11px] uppercase tracking-[0.18em] ${
                  status.kind === "success"
                    ? "text-nebula-success"
                    : status.kind === "error"
                      ? "text-nebula-danger"
                      : "text-nebula-on-dim"
                }`}
              >
                {status.kind === "success"
                  ? `→ Handshake OK · ${status.email} on the list.`
                  : status.kind === "error"
                    ? `! ${status.message}`
                    : "→ Awaiting input · stdin"}
              </p>
            </Card.Footer>
          </Form>
        </Card>
      </div>
    </Section>
  );
}
