import type { Conversation } from '../types';
import { dateLabel } from '../lib/format';

interface HistorySidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

export function HistorySidebar({
  conversations,
  activeId,
  open,
  onSelect,
  onNew,
  onDelete,
  onClose,
}: HistorySidebarProps) {
  const sorted = [...conversations].sort((left, right) => right.updatedAt - left.updatedAt);

  return (
    <>
      {open && (
        <button
          type="button"
          className="backdrop only-mobile"
          aria-label="Tutup riwayat"
          onClick={onClose}
        />
      )}
      <aside className={`win side ${open ? 'open' : ''}`.trim()}>
        <div className="win-title">
          <span>RIWAYAT</span>
          <span>{conversations.length}</span>
        </div>

        <div className="side-body">
          <button type="button" className="btn green full" onClick={onNew}>
            CHAT BARU
          </button>

          <ul className="threads">
            {sorted.map((conversation) => (
              <li
                key={conversation.id}
                className={`thread ${conversation.id === activeId ? 'active' : ''}`.trim()}
              >
                <button
                  type="button"
                  className="thread-main"
                  onClick={() => {
                    onSelect(conversation.id);
                    onClose();
                  }}
                >
                  <span className="thread-title">{conversation.title}</span>
                  <small>
                    {dateLabel(conversation.updatedAt)} · {conversation.messages.length} pesan
                  </small>
                </button>
                <button
                  type="button"
                  className="thread-del"
                  aria-label={`Hapus ${conversation.title}`}
                  onClick={() => onDelete(conversation.id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>

          <div className="legal">
            Sumber teks: Bulbapedia
            <br />
            CC BY-NC-SA 2.5
          </div>
        </div>
      </aside>
    </>
  );
}
