export function dateLabel(timestamp: number): string {
  const date = new Date(timestamp);
  const today = new Date();
  const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  const day = 86_400_000;

  if (timestamp >= startOfToday) return 'Hari ini';
  if (timestamp >= startOfToday - day) return 'Kemarin';
  return date.toLocaleDateString('id-ID', { day: 'numeric', month: 'short' });
}

export function padDex(dex: number): string {
  return String(dex).padStart(3, '0');
}

export function percent(score: number): number {
  return Math.max(0, Math.min(100, Math.round(score * 100)));
}
