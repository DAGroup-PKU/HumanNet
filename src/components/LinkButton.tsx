import type { AnchorHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "tertiary";
type Size = "sm" | "lg";

interface LinkButtonProps extends AnchorHTMLAttributes<HTMLAnchorElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
}

const variantClass: Record<Variant, string> = {
  primary:
    "bg-nebula-primary text-[var(--accent-foreground)] hover:bg-nebula-primary-soft focus-visible:outline-nebula-primary",
  tertiary:
    "border border-nebula-line bg-nebula-surface text-nebula-on hover:border-nebula-line-strong hover:bg-nebula-surface-2 focus-visible:outline-nebula-line-strong",
};

const sizeClass: Record<Size, string> = {
  sm: "h-9 px-3 text-xs",
  lg: "h-11 px-5 text-xs",
};

/**
 * Branded anchor styled like a primary action button. Used in place of
 * `<Button as="a">` because HeroUI v3's Button refuses to be cast to
 * an anchor (warns in console). Semantically this is more correct anyway —
 * a navigation target should be a link, not a button.
 */
export function LinkButton({
  variant = "primary",
  size = "lg",
  className = "",
  children,
  ...rest
}: LinkButtonProps) {
  return (
    <a
      {...rest}
      className={[
        "inline-flex items-center justify-center gap-2 rounded-md font-mono uppercase tracking-[0.18em]",
        "transition-colors duration-150 select-none",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline",
        variantClass[variant],
        sizeClass[size],
        className,
      ].join(" ")}
    >
      {children}
    </a>
  );
}
