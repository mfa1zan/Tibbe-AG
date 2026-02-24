import { useEffect, useRef } from 'react';
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

function ChatHistory({ messages, isTyping }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isTyping]);

  return (
    <section className="chat-history">
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