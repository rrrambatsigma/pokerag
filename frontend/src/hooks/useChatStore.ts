import { useCallback, useEffect, useRef, useState } from 'react';
import type { Message, Settings, UiState } from '../types';
import { modeFromSettings } from '../lib/mode';
import { emptyConversation, newId, titleFromText } from '../lib/conversations';
import { loadState, saveState } from '../lib/storage';
import { answerFor } from '../mock/data';

export function useChatStore() {
  const [state, setState] = useState<UiState>(() => loadState());
  const timers = useRef<Set<number>>(new Set());

  useEffect(() => {
    saveState(state);
  }, [state]);

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach((timer) => window.clearTimeout(timer));
      pending.clear();
    };
  }, []);

  const newChat = useCallback(() => {
    const conversation = emptyConversation();
    setState((prev) => ({
      ...prev,
      conversations: [conversation, ...prev.conversations],
      activeId: conversation.id,
    }));
    return conversation.id;
  }, []);

  const selectChat = useCallback((id: string) => {
    setState((prev) => ({ ...prev, activeId: id }));
  }, []);

  const deleteChat = useCallback((id: string) => {
    setState((prev) => {
      const conversations = prev.conversations.filter((conversation) => conversation.id !== id);

      if (conversations.length === 0) {
        const replacement = emptyConversation();
        return { ...prev, conversations: [replacement], activeId: replacement.id };
      }

      const activeId =
        prev.activeId === id
          ? [...conversations].sort((left, right) => right.updatedAt - left.updatedAt)[0].id
          : prev.activeId;

      return { ...prev, conversations, activeId };
    });
  }, []);

  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setState((prev) => ({ ...prev, settings: { ...prev.settings, ...patch } }));
  }, []);

  const send = useCallback(
    (text: string, settings: Settings) => {
      const trimmed = text.trim();
      if (!trimmed || modeFromSettings(settings) === 'off') return;

      const conversationId = state.activeId;
      if (!conversationId) return;

      const now = Date.now();
      const userMessage: Message = {
        id: newId(),
        role: 'user',
        text: trimmed,
        createdAt: now,
      };
      const assistantId = newId();
      const assistantMessage: Message = {
        id: assistantId,
        role: 'assistant',
        text: '',
        createdAt: now + 1,
        pending: true,
        settings: { ...settings },
      };

      setState((prev) => ({
        ...prev,
        conversations: prev.conversations.map((conversation) =>
          conversation.id !== conversationId
            ? conversation
            : {
                ...conversation,
                title:
                  conversation.messages.length === 0 && conversation.title === 'Chat baru'
                    ? titleFromText(trimmed)
                    : conversation.title,
                updatedAt: now,
                messages: [...conversation.messages, userMessage, assistantMessage],
              },
        ),
      }));

      const timer = window.setTimeout(() => {
        const answer = answerFor(trimmed);
        setState((prev) => ({
          ...prev,
          conversations: prev.conversations.map((conversation) =>
            conversation.id !== conversationId
              ? conversation
              : {
                  ...conversation,
                  updatedAt: Date.now(),
                  messages: conversation.messages.map((message) =>
                    message.id !== assistantId
                      ? message
                      : {
                          ...message,
                          pending: false,
                          text: answer.text,
                          citations: answer.citations,
                        },
                  ),
                },
          ),
        }));
        timers.current.delete(timer);
      }, 900);
      timers.current.add(timer);
    },
    [state.activeId],
  );

  return { state, newChat, selectChat, deleteChat, updateSettings, send };
}

export type ChatStore = ReturnType<typeof useChatStore>;
