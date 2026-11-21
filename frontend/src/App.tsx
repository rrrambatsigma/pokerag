import { useState } from 'react';
import { ChatPane } from './components/ChatPane';
import { HistorySidebar } from './components/HistorySidebar';
import { SettingsPage } from './components/SettingsPage';
import { SourceModal } from './components/SourceModal';
import { TopBar } from './components/TopBar';
import { useChatStore } from './hooks/useChatStore';
import { modeFromSettings } from './lib/mode';
import { SNAPSHOT } from './mock/data';
import type { Citation, Tab } from './types';

export default function App() {
  const { state, newChat, selectChat, deleteChat, updateSettings, send } = useChatStore();
  const [tab, setTab] = useState<Tab>('chat');
  const [citation, setCitation] = useState<Citation | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  const mode = modeFromSettings(state.settings);
  const active = state.conversations.find((conversation) => conversation.id === state.activeId) ?? null;

  const handleNew = () => {
    newChat();
    setTab('chat');
    setHistoryOpen(false);
  };

  const handleSend = (text: string) => {
    send(text, state.settings);
    setTab('chat');
  };

  return (
    <div className="app">
      <TopBar
        mode={mode}
        docs={SNAPSHOT.docs}
        tab={tab}
        onTab={setTab}
        onOpenHistory={() => setHistoryOpen(true)}
      />

      <div className="layout">
        <HistorySidebar
          conversations={state.conversations}
          activeId={state.activeId}
          open={historyOpen}
          onSelect={selectChat}
          onNew={handleNew}
          onDelete={deleteChat}
          onClose={() => setHistoryOpen(false)}
        />

        <main className="main">
          <section
            className={`page ${tab === 'chat' ? 'active' : ''}`.trim()}
            role="tabpanel"
            aria-label="Chat"
          >
            <ChatPane
              conversation={active}
              mode={mode}
              settings={state.settings}
              onSend={handleSend}
              onOpenCitation={setCitation}
            />
          </section>

          <section
            className={`page ${tab === 'settings' ? 'active' : ''}`.trim()}
            role="tabpanel"
            aria-label="Setelan"
          >
            <SettingsPage settings={state.settings} onChange={updateSettings} />
          </section>
        </main>
      </div>

      {citation && <SourceModal citation={citation} onClose={() => setCitation(null)} />}
    </div>
  );
}
