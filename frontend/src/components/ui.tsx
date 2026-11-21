import type { ButtonHTMLAttributes, ReactNode } from 'react';
import type { PokemonType } from '../types';

type ButtonVariant = 'green' | 'red' | 'orange' | 'blue' | 'plain' | '';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  full?: boolean;
  children: ReactNode;
}

export function PixelButton({
  variant = '',
  full = false,
  className = '',
  children,
  type = 'button',
  ...rest
}: ButtonProps) {
  const classes = ['btn', variant, full ? 'full' : '', className].filter(Boolean).join(' ');
  return (
    <button type={type} className={classes} {...rest}>
      {children}
    </button>
  );
}

interface SwitchProps {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}

export function PixelSwitch({ checked, label, onChange }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={`switch ${checked ? 'on' : ''}`.trim()}
      onClick={() => onChange(!checked)}
    >
      <span className="sr-only">{checked ? 'aktif' : 'mati'}</span>
    </button>
  );
}

interface ChipProps {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
}

export function PixelChip({ active, children, onClick }: ChipProps) {
  return (
    <button
      type="button"
      className={`chip ${active ? 'active' : ''}`.trim()}
      aria-pressed={active}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

interface StepperProps {
  value: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  label: string;
}

export function PixelStepper({ value, min = 1, max = 20, onChange, label }: StepperProps) {
  return (
    <span className="stepper" role="group" aria-label={label}>
      <PixelButton
        aria-label={`${label}: kurangi`}
        disabled={value <= min}
        onClick={() => onChange(Math.max(min, value - 1))}
      >
        −
      </PixelButton>
      <b aria-live="polite">{value}</b>
      <PixelButton
        aria-label={`${label}: tambah`}
        disabled={value >= max}
        onClick={() => onChange(Math.min(max, value + 1))}
      >
        +
      </PixelButton>
    </span>
  );
}

export function TypeBadge({ type }: { type: PokemonType }) {
  return <span className={`type type-${type}`}>{type.toUpperCase()}</span>;
}
