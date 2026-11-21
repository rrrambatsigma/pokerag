import type { HTMLAttributes, ReactNode } from 'react';

interface PanelProps extends HTMLAttributes<HTMLElement> {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
}

export function Panel({ title, meta, children, className = '', ...rest }: PanelProps) {
  return (
    <section className={`win ${className}`.trim()} {...rest}>
      <div className="win-title">
        <span>{title}</span>
        <span>{meta}</span>
      </div>
      {children}
    </section>
  );
}
