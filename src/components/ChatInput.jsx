import { useRef, useEffect } from 'react';
import './ChatInput.css';

function ChatInput({ value, onChange, onSend, onCancel, disabled, isGenerating, error }) {
  const inputRef = useRef(null);

  // Auto-focus when the form becomes interactive again.
  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSend(value);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSend(value);
    }
  };

  const handleInput = (event) => {
    const el = event.currentTarget;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  };

  return (
    <footer className="chat-input-footer">
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <textarea
            ref={inputRef}
            aria-label="Type your biomedical question"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            rows={1}
            placeholder="Type your biomedical question..."
            className="chat-input-field"
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
          {isGenerating ? (
            <button
              type="button"
              aria-label="Stop generation"
              onClick={onCancel}
              className="chat-cancel-button"
            >
              Stop
            </button>
          ) : null}
        </form>
        {error ? <p className="chat-input-error">{error}</p> : null}
      </div>
    </footer>
  );
}

export default ChatInput;