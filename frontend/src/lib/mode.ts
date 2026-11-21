import type { Mode, Settings } from '../types';

export function modeFromSettings(settings: Settings): Mode {
  if (settings.dense && settings.bm25) return 'hybrid';
  if (settings.dense) return 'dense';
  if (settings.bm25) return 'bm25';
  return 'off';
}

export const MODE_LABEL: Record<Mode, string> = {
  hybrid: 'HYBRID',
  dense: 'MAKNA',
  bm25: 'KATA KUNCI',
  off: 'MATI',
};

export function strategyLabel(strategy: Settings['strategy']): string {
  return strategy.toUpperCase();
}
