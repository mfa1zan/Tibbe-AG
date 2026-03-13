import { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';
import ReasoningPanel from './ReasoningPanel';
import './ChatBubble.css';

function ChatBubble({ message }) {
  const isUser = message.role === 'user';
  const isStatusMessage = message.variant === 'status';
  const displayText = message.content || (message.isStreaming ? '...' : '');
  const structuredFields = message.structuredFields;
  const hasStructuredFields =
    !isUser &&
    structuredFields &&
    typeof structuredFields === 'object' &&
    Object.values(structuredFields).some(
      (value) => value != null && String(value).trim().length > 0
    );
  const hasConfidence = typeof message.confidenceScore === 'number';
  const hasPaths = Number.isFinite(message.graphPathsUsed);
  const hasEvidenceStrength = typeof message.evidenceStrength === 'string' && message.evidenceStrength.length > 0;
  const showMeta = !isUser && (hasConfidence || hasPaths || hasEvidenceStrength);

  if (isStatusMessage) {
    return (
      <article className="chat-status-row fade-in" aria-live="polite">
        <p className="chat-status-text">{displayText}</p>
      </article>
    );
  }

  return (
    <article
      className={`chat-bubble-row ${isUser ? 'chat-bubble-row-user' : 'chat-bubble-row-bot'} ${
        isUser ? 'message-send-pop' : 'fade-in'
      }`}
    >
      <div
        className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-bot'}`}
        style={isUser ? { backgroundColor: 'var(--primary-color)' } : undefined}
      >
        {isUser ? (
          <p className="chat-message-text">{displayText}</p>
        ) : message.isStreaming && !message.content ? (
          <div className="chat-streaming-indicator" aria-label="Assistant is replying">
            <span className="chat-streaming-dot chat-streaming-dot-delay-1" />
            <span className="chat-streaming-dot chat-streaming-dot-delay-2" />
            <span className="chat-streaming-dot" />
          </div>
        ) : (
          <div className="chat-markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
              {displayText}
            </ReactMarkdown>
          </div>
        )}
        {showMeta ? (
          <p className="chat-provenance">
            Evidence: {hasEvidenceStrength ? message.evidenceStrength : 'weak'}
            {hasPaths ? ` • Paths: ${message.graphPathsUsed}` : ''}
            {hasConfidence ? ` • Confidence: ${(message.confidenceScore * 100).toFixed(0)}%` : ''}
          </p>
        ) : null}
        {hasStructuredFields ? <StructuredFieldsCard fields={structuredFields} /> : null}
        {!isUser && message.reasoningTrace && (
          <ReasoningPanel trace={message.reasoningTrace} />
        )}
      </div>
    </article>
  );
}

function StructuredFieldsCard({ fields }) {
  const entries = Object.entries(fields)
    .filter(([, value]) => value != null && String(value).trim().length > 0)
    .slice(0, 8);

  if (entries.length === 0) return null;

  return (
    <section className="structured-fields-card" aria-label="Structured evidence fields">
      <h4 className="structured-fields-title">Structured Evidence</h4>
      <dl className="structured-fields-grid">
        {entries.map(([key, value]) => (
          <div key={key} className="structured-field-item">
            <dt className="structured-field-key">{formatFieldKey(key)}</dt>
            <dd className="structured-field-value">{formatFieldValue(value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function formatFieldKey(value) {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatFieldValue(value) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }

  return String(value);
}

export default memo(ChatBubble);