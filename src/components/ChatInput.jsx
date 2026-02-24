function ChatInput({ value, onChange, onSend, disabled, error }) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSend(value);
  };

  return (
    <footer className="fixed inset-x-0 bottom-0 border-t border-slate-200 bg-white/95 shadow-[0_-6px_20px_rgba(15,23,42,0.08)] backdrop-blur">
      <div className="mx-auto w-full max-w-4xl px-4 py-3">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Type your biomedical question..."
            className="flex-1 rounded-xl border border-slate-300 px-4 py-2.5 text-sm text-slate-900 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200 disabled:bg-slate-100"
            disabled={disabled}
          />
          <button
            type="submit"
            disabled={disabled || !value.trim()}
            className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300"
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