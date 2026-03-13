import ThemeToggle from '../components/ThemeToggle';

function SettingsPage({ messages = [], onClearConversation }) {
  const persistedMessages = messages
    .filter((message) => !message.isStreaming && message.role === 'user')
    .slice()
    .reverse();

  return (
    <section className="app-route-panel">
      <h2 className="app-route-title">Display Settings</h2>
      <p className="app-route-helper">
        Customize theme mode, font family, and primary color for your chat workspace.
      </p>
      <ThemeToggle />

      <div className="app-settings-history">
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
    </section>
  );
}

export default SettingsPage;
