function ChatBubble({ message }) {
  const isUser = message.role === 'user';
  const showProvenance = !isUser && Array.isArray(message.provenance) && message.provenance.length > 0;

  return (
    <article className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? 'rounded-br-sm bg-blue-600 text-white'
            : 'rounded-bl-sm bg-slate-200 text-slate-900'
        }`}
      >
        {message.content}
        {showProvenance ? (
          <p className="mt-2 border-t border-slate-300/60 pt-2 text-xs text-slate-700">
            Based on KG: {message.provenance.join(', ')}
          </p>
        ) : null}
      </div>
    </article>
  );
}

export default ChatBubble;