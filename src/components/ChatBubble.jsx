function ChatBubble({ message }) {
  const isUser = message.role === 'user';

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
      </div>
    </article>
  );
}

export default ChatBubble;