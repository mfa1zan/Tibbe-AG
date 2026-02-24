import './ChatBubble.css';

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlightEntities(content, provenance) {
  if (!Array.isArray(provenance) || provenance.length === 0) {
    return [{ text: content, match: null }];
  }

  const sortedTerms = [...provenance].sort((first, second) => second.length - first.length);
  const pattern = new RegExp(`(${sortedTerms.map((term) => escapeRegExp(term)).join('|')})`, 'gi');
  const pieces = content.split(pattern).filter(Boolean);

  return pieces.map((piece) => {
    const matched = provenance.find((term) => term.toLowerCase() === piece.toLowerCase()) || null;
    return { text: piece, match: matched };
  });
}

function ChatBubble({ message }) {
  const isUser = message.role === 'user';
  const showProvenance = !isUser && Array.isArray(message.provenance) && message.provenance.length > 0;
  const tokens = highlightEntities(message.content, message.provenance || []);

  return (
    <article
      className={`chat-bubble-row ${isUser ? 'chat-bubble-row-user' : 'chat-bubble-row-bot'} ${
        !isUser ? 'fade-in' : ''
      }`}
    >
      <div
        className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-bot'}`}
        style={isUser ? { backgroundColor: 'var(--primary-color)' } : undefined}
      >
        {tokens.map((token, index) => {
          if (!token.match || isUser) {
            return <span key={`${token.text}-${index}`}>{token.text}</span>;
          }

          const related = (message.provenance || []).filter((node) => node !== token.match).slice(0, 4);

          return (
            <span key={`${token.text}-${index}`} className="group kg-token-wrap">
              <span className="kg-token-text">{token.text}</span>
              <span className="kg-token-popup">
                <strong className="kg-token-popup-title">{token.match}</strong>
                {related.length > 0 ? (
                  <span className="kg-token-popup-content">Connected: {related.join(', ')}</span>
                ) : (
                  <span className="kg-token-popup-content">No connected nodes available.</span>
                )}
              </span>
            </span>
          );
        })}
        {showProvenance ? (
          <p className="chat-provenance">
            Based on KG: {message.provenance.join(', ')}
          </p>
        ) : null}
      </div>
    </article>
  );
}

export default ChatBubble;