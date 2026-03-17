import { useMemo, useState } from 'react';
import ThemeToggle from '../components/ThemeToggle';

const SETTINGS_TABS = {
  KNOWLEDGE: 'knowledge',
  DISPLAY: 'display',
  HISTORY: 'history'
};

function SettingsPage({
  messages = [],
  strictMode = false,
  onStrictModeChange,
  onClearConversation
}) {
  const [activeTab, setActiveTab] = useState(SETTINGS_TABS.KNOWLEDGE);

  const persistedMessages = useMemo(
    () =>
      messages
        .filter((message) => !message.isStreaming && message.role === 'user')
        .slice()
        .reverse(),
    [messages]
  );

  return (
    <section className="app-route-panel">
      <h2 className="app-route-title">Settings</h2>

      <div className="app-settings-layout">
        <aside className="app-settings-sidebar" role="tablist" aria-label="Settings sections">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === SETTINGS_TABS.KNOWLEDGE}
            className={`app-settings-sidebar-tab ${activeTab === SETTINGS_TABS.KNOWLEDGE ? 'app-settings-sidebar-tab-active' : ''}`}
            onClick={() => setActiveTab(SETTINGS_TABS.KNOWLEDGE)}
          >
            Knowledge
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === SETTINGS_TABS.DISPLAY}
            className={`app-settings-sidebar-tab ${activeTab === SETTINGS_TABS.DISPLAY ? 'app-settings-sidebar-tab-active' : ''}`}
            onClick={() => setActiveTab(SETTINGS_TABS.DISPLAY)}
          >
            Display
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === SETTINGS_TABS.HISTORY}
            className={`app-settings-sidebar-tab ${activeTab === SETTINGS_TABS.HISTORY ? 'app-settings-sidebar-tab-active' : ''}`}
            onClick={() => setActiveTab(SETTINGS_TABS.HISTORY)}
          >
            Chat History
          </button>
        </aside>

        <div className="app-settings-content">
          {activeTab === SETTINGS_TABS.KNOWLEDGE ? (
            <div className="app-settings-panel" role="tabpanel">
              <h3 className="app-route-title">Knowledge Mode</h3>
              <p className="app-route-helper">
                Choose whether the model can answer from its own knowledge when Neo4j has no relevant evidence.
              </p>

              <div className="app-settings-row">
                <div>
                  <p className="app-settings-label">Allow model fallback knowledge</p>
                  <p className="app-settings-subtext">
                    When disabled, the assistant stays DB-only and returns a clear no-data message.
                  </p>
                </div>
                <button
                  type="button"
                  className={`app-toggle-button ${strictMode ? '' : 'app-toggle-button-active'}`}
                  onClick={() => onStrictModeChange?.(!strictMode)}
                  aria-pressed={!strictMode}
                >
                  {strictMode ? 'Off' : 'On'}
                </button>
              </div>

              <p className="app-route-helper">
                DB-only mode is currently: <strong>{strictMode ? 'ON' : 'OFF'}</strong>
              </p>
            </div>
          ) : null}

          {activeTab === SETTINGS_TABS.DISPLAY ? (
            <div className="app-settings-panel" role="tabpanel">
              <h3 className="app-route-title">Display Settings</h3>
              <p className="app-route-helper">
                Customize theme mode, font family, and primary color for your chat workspace.
              </p>
              <ThemeToggle />
            </div>
          ) : null}

          {activeTab === SETTINGS_TABS.HISTORY ? (
            <div className="app-settings-panel app-settings-history" role="tabpanel">
              <div className="app-route-header">
                <h3 className="app-route-title">Conversation History</h3>
                <button type="button" className="app-clear-button" onClick={onClearConversation}>
                  Clear Chat
                </button>
              </div>

              {persistedMessages.length === 0 ? (
                <p className="app-route-empty">No user messages found yet.</p>
              ) : (
                <ul className="app-history-list">
                  {persistedMessages.map((message) => (
                    <li key={message.id} className="app-history-item">
                      {message.content}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default SettingsPage;
