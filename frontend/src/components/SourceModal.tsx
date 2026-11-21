import { useEffect, useRef } from 'react';
import type { Citation } from '../types';
import { padDex, percent } from '../lib/format';
import { TypeBadge } from './ui';

interface SourceModalProps {
  citation: Citation;
  onClose: () => void;
}

export function SourceModal({ citation, onClose }: SourceModalProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal win"
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="win-title">
          <span>POKéDEX ENTRY</span>
          <button ref={closeRef} type="button" className="close" onClick={onClose} aria-label="Tutup">
            ×
          </button>
        </div>

        <div className="modal-body">
          <div className="entry-head">
            <div>
              <h2 id="source-title">{citation.title}</h2>
              <p className="meta">
                No. {padDex(citation.dex)} · {citation.headingPath.join(' › ')}
              </p>
              <div className="types">
                {citation.types.map((type) => (
                  <TypeBadge key={type} type={type} />
                ))}
              </div>
            </div>
            <span className={`bar big ${citation.score >= 0.85 ? 'g' : 'y'}`}>
              <i style={{ width: `${percent(citation.score)}%` }} />
            </span>
          </div>

          <blockquote className="excerpt">{citation.excerpt}</blockquote>

          <div className="kv">
            <span>Skor</span>
            <span>{percent(citation.score)}%</span>
            <span>Chunk</span>
            <span>{citation.chunkId}</span>
          </div>

          <a className="btn blue full" href={citation.url} target="_blank" rel="noreferrer">
            BUKA DI BULBAPEDIA
          </a>

          <p className="attribution">Bulbapedia · CC BY-NC-SA 2.5</p>
        </div>
      </div>
    </div>
  );
}
