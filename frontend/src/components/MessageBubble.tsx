import type { Citation, Message } from '../types';
import { SourceList } from './Citations';

interface MessageBubbleProps {
  message: Message;
  onOpenCitation: (citation: Citation) => void;
}

export function MessageBubble({ message, onOpenCitation }: MessageBubbleProps) {
  if (message.role === 'user') {
    return <div className="bubble user">{message.text}</div>;
  }

  const citations = message.citations ?? [];

  return (
    <div className="assistant-block">
      <div className={`bubble bot ${message.pending ? 'pending' : 'done'}`.trim()}>
        <div className="speaker">POKéDEX</div>
        {message.pending ? (
          <span className="dots" aria-label="Menyusun jawaban">
            <i />
            <i />
            <i />
          </span>
        ) : (
          <p className="answer">{message.text}</p>
        )}
        {!message.pending && citations.length > 0 && (
          <div className="cites">
            {citations.map((citation, index) => (
              <button
                key={citation.chunkId}
                type="button"
                className={`cite ${index % 2 === 1 ? 'c2' : ''}`.trim()}
                onClick={() => onOpenCitation(citation)}
                aria-label={`Sumber ${citation.n}: ${citation.title}`}
              >
                {citation.n}
              </button>
            ))}
          </div>
        )}
      </div>

      {!message.pending && citations.length > 0 && (
        <SourceList citations={citations} onOpen={onOpenCitation} />
      )}
    </div>
  );
}
