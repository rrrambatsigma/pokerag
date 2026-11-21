import { useEffect, useRef } from 'react';
import type { Citation, Conversation, Mode, Settings } from '../types';
import { MODE_LABEL, strategyLabel } from '../lib/mode';
import { SUGGESTIONS } from '../mock/data';
import { Composer } from './Composer';
import { MessageBubble } from './MessageBubble';

interface ChatPaneProps {
  conversation: Conversation | null;
  mode: Mode;
  settings: Settings;
  onSend: (text: string) => void;
  onOpenCitation: (citation: Citation) => void;
}

export function ChatPane({
  conversation,
  mode,
  settings,
  onSend,
  onOpenCitation,
}: ChatPaneProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const messages = conversation?.messages ?? [];
  const disabled = mode === 'off';
  const lastMessage = messages[messages.length - 1];

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [conversation?.id, messages.length, lastMessage?.pending]);

  return (
    <div className="win chat">
      <div className="win-title">
        <span>CHAT</span>
        <span>
          {MODE_LABEL[mode]} · {strategyLabel(settings.strategy)}
        </span>
      </div>

      <div className="messages" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty">
            <p className="empty-title">Pertanyaan liar muncul!</p>
            <p className="empty-copy">
              Tulis pertanyaan di bawah, atau mulai dari salah satu contoh ini.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="chip"
                  disabled={disabled}
                  onClick={() => onSend(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} onOpenCitation={onOpenCitation} />
          ))
        )}
        <div ref={endRef} />
      </div>

      <Composer disabled={disabled} onSend={onSend} />
    </div>
  );
}
