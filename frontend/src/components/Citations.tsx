import type { Citation } from '../types';
import { padDex, percent } from '../lib/format';
import { TypeBadge } from './ui';

interface SourceListProps {
  citations: Citation[];
  onOpen: (citation: Citation) => void;
}

export function SourceList({ citations, onOpen }: SourceListProps) {
  return (
    <div className="sources">
      <h4>POKéDEX · {citations.length} ENTRI</h4>
      {citations.map((citation, index) => (
        <button
          key={citation.chunkId}
          type="button"
          className="src"
          onClick={() => onOpen(citation)}
          aria-label={`Buka sumber ${citation.n}: ${citation.title}`}
        >
          <span className={`num ${index % 2 === 1 ? 'o' : ''}`.trim()}>{citation.n}</span>
          <span className="src-main">
            <span className="src-title">{citation.title}</span>
            <span className="meta">
              No. {padDex(citation.dex)} · {citation.headingPath.join(' › ')}
            </span>
            <span className="types">
              {citation.types.map((type) => (
                <TypeBadge key={type} type={type} />
              ))}
            </span>
          </span>
          <span className={`bar ${citation.score >= 0.85 ? 'g' : 'y'}`}>
            <i style={{ width: `${percent(citation.score)}%` }} />
          </span>
        </button>
      ))}
    </div>
  );
}
