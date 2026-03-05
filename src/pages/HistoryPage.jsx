function HistoryPage({ messages, onClearConversation }) {
  const persistedMessages = messages
    .filter((message) => !message.isStreaming && message.role === 'user')
    .slice()
    .reverse();

  return (
    <section className="app-route-panel">
      <div className="app-route-header">
        <h2 className="app-route-title">Conversation History</h2>
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
    </section>
  );
}

export default HistoryPage;
