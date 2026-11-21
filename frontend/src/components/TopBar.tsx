import { MODE_LABEL } from '../lib/mode';
import type { Mode, Tab } from '../types';

interface TopBarProps {
  mode: Mode;
  docs: number;
  tab: Tab;
  onTab: (tab: Tab) => void;
  onOpenHistory: () => void;
}

export function TopBar({ mode, docs, tab, onTab, onOpenHistory }: TopBarProps) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="glyph" aria-hidden="true" />
        POKéRAG
      </div>

      <div className="tabs" role="tablist" aria-label="Bagian aplikasi">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'chat'}
          className={`tab ${tab === 'chat' ? 'active' : ''}`.trim()}
          onClick={() => onTab('chat')}
        >
          CHAT
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'settings'}
          className={`tab ${tab === 'settings' ? 'active' : ''}`.trim()}
          onClick={() => onTab('settings')}
        >
          SETELAN
        </button>
      </div>

      <button type="button" className="btn blue only-mobile" onClick={onOpenHistory}>
        RIWAYAT
      </button>

      <div className="spacer" />

      <div className={`pill ${mode}`}>{MODE_LABEL[mode]}</div>
      <div className="pill plain">{docs} POKéMON</div>
    </header>
  );
}
