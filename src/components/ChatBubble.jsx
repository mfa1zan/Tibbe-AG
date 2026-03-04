import ReasoningPanel from './ReasoningPanel';
import './ChatBubble.css';

function ChatBubble({ message }) {
  const isUser = message.role === 'user';
  const displayText = message.content || (message.isStreaming ? '...' : '');
  const hasConfidence = typeof message.confidenceScore === 'number';
  const hasPaths = Number.isFinite(message.graphPathsUsed);
  const hasEvidenceStrength = typeof message.evidenceStrength === 'string' && message.evidenceStrength.length > 0;
  const showMeta = !isUser && (hasConfidence || hasPaths || hasEvidenceStrength);

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
        <p className="chat-message-text">{displayText}</p>
        {showMeta ? (
          <p className="chat-provenance">
            Evidence: {hasEvidenceStrength ? message.evidenceStrength : 'weak'}
            {hasPaths ? ` • Paths: ${message.graphPathsUsed}` : ''}
            {hasConfidence ? ` • Confidence: ${(message.confidenceScore * 100).toFixed(0)}%` : ''}
          </p>
        ) : null}
        {!isUser && message.reasoningTrace && (
          <ReasoningPanel trace={message.reasoningTrace} />
        )}
      </div>
    </article>
  );
}

export default ChatBubble;