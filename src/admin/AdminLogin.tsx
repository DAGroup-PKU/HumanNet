import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Input, Label, TextField } from "@heroui/react";
import { adminApi, HttpError, setToken } from "./api";

export function AdminLogin() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await adminApi.login(username, password);
      setToken(res.token);
      navigate("/admin", { replace: true });
    } catch (e) {
      if (e instanceof HttpError) {
        if (e.status === 401) setError("Wrong username or password.");
        else if (e.status === 429) setError("Too many attempts — wait a minute.");
        else setError(`Login failed (HTTP ${e.status}).`);
      } else {
        setError("Network error — is the API running?");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="grid min-h-dvh place-items-center px-6">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm rounded-lg border border-nebula-line bg-nebula-surface p-8 shadow-[0_30px_80px_-30px_rgba(0,0,0,0.7)]"
      >
        <div className="mb-1 flex items-center gap-3">
          <span
            aria-hidden="true"
            className="grid h-8 w-8 place-items-center rounded-sm bg-nebula-primary/10 ring-1 ring-nebula-primary/40"
          >
            <span className="block h-2.5 w-2.5 rounded-full bg-nebula-primary" />
          </span>
          <span className="font-display text-lg text-nebula-on">
            Project <span className="text-nebula-primary">Nebula</span>
          </span>
        </div>
        <h1 className="font-display text-2xl text-nebula-on">Admin sign-in</h1>
        <p className="mt-2 text-sm text-nebula-on-muted">
          Token-based session. Editor only — public site is unaffected by
          mistakes here.
        </p>

        <div className="mt-6 flex flex-col gap-4">
          <TextField name="username" isRequired value={username} onChange={setUsername} autoFocus>
            <Label className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
              Username
            </Label>
            <Input variant="secondary" placeholder="admin" autoComplete="username" />
          </TextField>
          <TextField
            name="password"
            type="password"
            isRequired
            value={password}
            onChange={setPassword}
          >
            <Label className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
              Password
            </Label>
            <Input variant="secondary" placeholder="••••••••" autoComplete="current-password" />
          </TextField>
        </div>

        <Button
          type="submit"
          className="mt-6 w-full font-mono text-xs uppercase tracking-[0.18em]"
          isDisabled={submitting}
        >
          {submitting ? "Signing in…" : "Sign in"}
        </Button>

        <p
          role="status"
          aria-live="polite"
          className={`mt-4 min-h-[1.25rem] font-mono text-[11px] uppercase tracking-[0.18em] ${
            error ? "text-nebula-danger" : "text-nebula-on-dim"
          }`}
        >
          {error ?? "→ Awaiting credentials"}
        </p>
      </form>
    </div>
  );
}
