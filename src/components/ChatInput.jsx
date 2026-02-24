import './ChatInput.css';

function ChatInput({ value, onChange, onSend, disabled, error }) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSend(value);
  };

  return (
    <footer className="chat-input-footer">
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            aria-label="Type your biomedical question"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Type your biomedical question..."
            className="chat-input-field"
            disabled={disabled}
          />
          <button
            type="submit"
            aria-label="Send message"
            disabled={disabled || !value.trim()}
            className="chat-send-button"
            style={{
              backgroundColor: disabled || !value.trim() ? undefined : 'var(--primary-color)'
            }}
          >
            Send
          </button>
        </form>
        {error ? <p className="chat-input-error">{error}</p> : null}
      </div>
    </footer>
  );
}

export default ChatInput;