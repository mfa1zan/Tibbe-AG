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
    <article className={`flex ${isUser ? 'justify-end' : 'justify-start'} ${!isUser ? 'fade-in' : ''}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? 'rounded-br-sm text-white'
            : 'rounded-bl-sm bg-slate-200 text-slate-900 transition-colors dark:bg-slate-800 dark:text-slate-100'
        }`}
        style={isUser ? { backgroundColor: 'var(--primary-color)' } : undefined}
      >
        {tokens.map((token, index) => {
          if (!token.match || isUser) {
            return <span key={`${token.text}-${index}`}>{token.text}</span>;
          }

          const related = (message.provenance || []).filter((node) => node !== token.match).slice(0, 4);

          return (
            <span key={`${token.text}-${index}`} className="group relative inline-block cursor-help font-medium">
              <span className="rounded px-0.5 text-[var(--primary-color)] underline decoration-dotted">
                {token.text}
              </span>
              <span className="pointer-events-none absolute left-0 top-full z-20 mt-2 hidden w-56 rounded-lg border border-slate-300 bg-white p-2 text-xs text-slate-800 shadow-lg group-hover:block dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100">
                <strong className="block text-[var(--primary-color)]">{token.match}</strong>
                {related.length > 0 ? (
                  <span className="mt-1 block">Connected: {related.join(', ')}</span>
                ) : (
                  <span className="mt-1 block">No connected nodes available.</span>
                )}
              </span>
            </span>
          );
        })}
        {showProvenance ? (
          <p className="mt-2 border-t border-slate-300/60 pt-2 text-xs text-slate-700 dark:border-slate-600/70 dark:text-slate-300">
            Based on KG: {message.provenance.join(', ')}
          </p>
        ) : null}
      </div>
    </article>
  );
}

export default ChatBubble;