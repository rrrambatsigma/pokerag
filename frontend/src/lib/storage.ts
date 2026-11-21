import type { UiState } from '../types';
import { DEFAULT_SETTINGS, seedConversations } from '../mock/data';

const KEY = 'pokerag.ui.v1';

export function initialState(): UiState {
  const conversations = seedConversations();
  return {
    version: 1,
    conversations,
    activeId: conversations[0]?.id ?? null,
    settings: { ...DEFAULT_SETTINGS },
  };
}

export function loadState(): UiState {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return initialState();
    const parsed = JSON.parse(raw) as Partial<UiState>;
    if (parsed.version !== 1 || !Array.isArray(parsed.conversations)) return initialState();

    const conversations = parsed.conversations.filter(
      (conversation) => conversation && typeof conversation.id === 'string' && Array.isArray(conversation.messages),
    );
    if (conversations.length === 0) return initialState();
    const settings = { ...DEFAULT_SETTINGS, ...(parsed.settings ?? {}) };
    const activeId =
      parsed.activeId && conversations.some((conversation) => conversation.id === parsed.activeId)
        ? parsed.activeId
        : (conversations[0]?.id ?? null);

    return { version: 1, conversations, activeId, settings };
  } catch {
    return initialState();
  }
}

export function saveState(state: UiState): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // A full or blocked localStorage should not break the chat.
  }
}

export function clearState(): void {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
