import type { ReactNode } from "react";

interface FieldProps {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Plain controlled label/input wrapper used across the admin editors.
 *  Doesn't go through HeroUI's TextField because the editors need full
 *  control over `value` / `onChange` for nested arrays. */
export function Field({ label, hint, children, className = "" }: FieldProps) {
  return (
    <label className={`flex flex-col gap-2 ${className}`}>
      <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-muted">
        {label}
      </span>
      {children}
      {hint && (
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-nebula-on-dim">
          {hint}
        </span>
      )}
    </label>
  );
}

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "value" | "onChange"> {
  value: string | number;
  onChange: (next: string) => void;
}

export function TextInput({ value, onChange, className = "", ...rest }: InputProps) {
  return (
    <input
      {...rest}
      value={String(value)}
      onChange={(e) => onChange(e.target.value)}
      className={`rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 text-sm text-nebula-on placeholder:text-nebula-on-dim focus:border-nebula-primary focus:outline-none focus:ring-1 focus:ring-nebula-primary/40 ${className}`}
    />
  );
}

interface AreaProps extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "value" | "onChange"> {
  value: string;
  onChange: (next: string) => void;
}

export function TextArea({ value, onChange, className = "", rows = 4, ...rest }: AreaProps) {
  return (
    <textarea
      {...rest}
      value={value}
      rows={rows}
      onChange={(e) => onChange(e.target.value)}
      className={`rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 text-sm text-nebula-on placeholder:text-nebula-on-dim focus:border-nebula-primary focus:outline-none focus:ring-1 focus:ring-nebula-primary/40 ${className}`}
    />
  );
}

interface SelectProps<T extends string> extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "value" | "onChange"> {
  value: T;
  options: { id: T; label: string }[];
  onChange: (next: T) => void;
}

export function NativeSelect<T extends string>({
  value,
  options,
  onChange,
  className = "",
  ...rest
}: SelectProps<T>) {
  return (
    <select
      {...rest}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className={`rounded-sm border border-nebula-line bg-nebula-surface px-3 py-2 text-sm text-nebula-on focus:border-nebula-primary focus:outline-none ${className}`}
    >
      {options.map((o) => (
        <option key={o.id} value={o.id}>
          {o.label}
        </option>
      ))}
    </select>
  );
}
