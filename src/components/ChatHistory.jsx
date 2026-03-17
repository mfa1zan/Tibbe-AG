import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { Virtuoso } from 'react-virtuoso';
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
  const virtuosoRef = useRef(null);
  const isNearBottomRef = useRef(true);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const hasStreamingMessage = messages.some((message) => message.isStreaming);

  /** Track whether the user has scrolled away from the bottom. */
  const handleBottomStateChange = useCallback((isAtBottom) => {
    isNearBottomRef.current = isAtBottom;
    setShowJumpToBottom(!isAtBottom);
  }, []);

  const handleJumpToBottom = useCallback(() => {
    isNearBottomRef.current = true;
    setShowJumpToBottom(false);
    virtuosoRef.current?.scrollToIndex({
      index: Math.max(0, messages.length - 1),
      align: 'end',
      behavior: 'smooth'
    });
  }, [messages.length]);

  /** Auto-scroll only when user is near the bottom. */
  useEffect(() => {
    if (isNearBottomRef.current) {
      virtuosoRef.current?.scrollToIndex({
        index: Math.max(0, messages.length - 1),
        align: 'end',
        behavior: 'smooth'
      });
    }
  }, [messages, isTyping]);

  /** Always scroll to bottom when the user sends a new message (last msg is "user"). */
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    if (lastMsg?.role === 'user') {
      isNearBottomRef.current = true;
      virtuosoRef.current?.scrollToIndex({
        index: Math.max(0, messages.length - 1),
        align: 'end',
        behavior: 'smooth'
      });
    }
  }, [messages]);

  return (
    <section className="chat-history">
      <Virtuoso
        ref={virtuosoRef}
        className="chat-history-virtuoso"
        data={messages}
        overscan={220}
        atBottomThreshold={SCROLL_THRESHOLD}
        atBottomStateChange={handleBottomStateChange}
        followOutput={(isAtBottom) => (isAtBottom ? 'smooth' : false)}
        itemContent={(_, message) => <ChatBubble message={message} />}
        components={{
          Footer: () => (isTyping && !hasStreamingMessage ? <TypingIndicator /> : null)
        }}
      />

      {showJumpToBottom ? (
        <button
          type="button"
          className="chat-jump-to-bottom"
          aria-label="Jump to latest messages"
          onClick={handleJumpToBottom}
        >
          ↓
        </button>
      ) : null}
    </section>
  );
}

export default memo(ChatHistory);