import type { Conversation } from '../types';

export function newId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function emptyConversation(): Conversation {
  const now = Date.now();
  return {
    id: newId(),
    title: 'Chat baru',
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function titleFromText(text: string): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= 32) return clean;
  return `${clean.slice(0, 31).trimEnd()}…`;
}
