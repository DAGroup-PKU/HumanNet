import type { ReactNode } from "react";

interface SectionProps {
  id: string;
  eyebrow?: string;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * Vertical section primitive — gives every page section a consistent rhythm
 * (eyebrow → headline → description → body) and the same outer padding.
 */
export function Section({ id, eyebrow, title, description, children, className = "" }: SectionProps) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-title`}
      className={`relative scroll-mt-24 px-6 py-20 sm:px-10 sm:py-24 lg:px-16 lg:py-32 ${className}`}
    >
      <div className="mx-auto w-full max-w-7xl">
        <header className="mb-10 max-w-3xl sm:mb-14">
          {eyebrow && <div className="eyebrow mb-4">{eyebrow}</div>}
          <h2
            id={`${id}-title`}
            className="text-balance text-3xl font-medium leading-[1.05] tracking-tight text-nebula-on sm:text-4xl lg:text-5xl"
          >
            {title}
          </h2>
          {description && (
            <p className="mt-5 max-w-2xl text-base leading-relaxed text-nebula-on-muted sm:text-lg">
              {description}
            </p>
          )}
        </header>
        {children}
      </div>
    </section>
  );
}
