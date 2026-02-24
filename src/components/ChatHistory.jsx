import { useEffect, useRef } from 'react';
import ChatBubble from './ChatBubble';

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-sm bg-slate-200 px-4 py-3 text-slate-700">
        <div className="flex items-center gap-1">
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-500 [animation-delay:-0.3s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-500 [animation-delay:-0.15s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-500" />
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
    <section className="flex-1 overflow-y-auto px-4 py-4 pb-32">
      <div className="space-y-3">
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