import { useCallback, useEffect, useRef } from 'react';
import ChatBubble from './ChatBubble';
import './ChatHistory.css';

function TypingIndicator() {
  return (
    <div className="typing-wrap fade-in">
      <div className="typing-bubble">
        <div className="typing-dots">
          <span className="typing-dot typing-dot-delay-1" />
          <span className="typing-dot typing-dot-delay-2" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

/** Threshold (px) from the bottom within which we consider the user "at bottom". */
const SCROLL_THRESHOLD = 120;

function ChatHistory({ messages, isTyping }) {
  const containerRef = useRef(null);
  const endRef = useRef(null);
  const isNearBottomRef = useRef(true);

  /** Track whether the user has scrolled away from the bottom. */
  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceFromBottom <= SCROLL_THRESHOLD;
  }, []);

  /** Auto-scroll only when user is near the bottom. */
  useEffect(() => {
    if (isNearBottomRef.current) {
      endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages, isTyping]);

  /** Always scroll to bottom when the user sends a new message (last msg is "user"). */
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    if (lastMsg?.role === 'user') {
      isNearBottomRef.current = true;
      endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages]);

  return (
    <section className="chat-history" ref={containerRef} onScroll={handleScroll}>
      <div className="chat-history-inner">
        {messages.map((message) => (
          <ChatBubble key={message.id} message={message} />
        ))}

        {isTyping ? <TypingIndicator /> : null}
        <div ref={endRef} />
      </div>
    </section>
  );
}

export default ChatHistory;