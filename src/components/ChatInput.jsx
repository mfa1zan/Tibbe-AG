function ChatInput({ value, onChange, onSend, disabled, error }) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSend(value);
  };

  return (
    <footer className="fixed inset-x-0 bottom-0 border-t border-slate-200 bg-white/95 shadow-[0_-6px_20px_rgba(15,23,42,0.08)] backdrop-blur transition-colors dark:border-slate-800 dark:bg-slate-900/95">
      <div className="mx-auto w-full max-w-4xl px-4 py-3">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            aria-label="Type your biomedical question"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Type your biomedical question..."
            className="flex-1 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm text-slate-900 outline-none transition focus:border-[var(--primary-color)] focus:ring-2 focus:ring-[var(--primary-color)]/25 disabled:bg-slate-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:disabled:bg-slate-800"
            disabled={disabled}
          />
          <button
            type="submit"
            aria-label="Send message"
            disabled={disabled || !value.trim()}
            className="rounded-xl px-4 py-2.5 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:bg-slate-400 dark:disabled:bg-slate-600"
            style={{
              backgroundColor: disabled || !value.trim() ? undefined : 'var(--primary-color)'
            }}
          >
            Send
          </button>
        </form>
        {error ? <p className="mt-2 text-sm text-red-600">{error}</p> : null}
      </div>
    </footer>
  );
}

export default ChatInput;