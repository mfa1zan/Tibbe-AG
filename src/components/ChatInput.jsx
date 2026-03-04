import { useRef, useEffect } from 'react';
import './ChatInput.css';

function ChatInput({ value, onChange, onSend, onCancel, disabled, isGenerating, error }) {
  const inputRef = useRef(null);

  // Auto-focus the input field whenever it becomes enabled or value is cleared
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, value]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSend(value);
  };

  return (
    <footer className="chat-input-footer">
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            ref={inputRef}
            type="text"
            aria-label="Type your biomedical question"
            value={value}
            onChange={(event) => onChange(event.target.value)}
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