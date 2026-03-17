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
          <div className="chat-input-shell">
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
            {isGenerating ? (
              <button
                type="button"
                aria-label="Stop generation"
                onClick={onCancel}
                className="chat-inline-action-button chat-inline-action-stop"
              >
                ⏹
              </button>
            ) : (
              <button
                type="submit"
                aria-label="Send message"
                disabled={disabled || !value.trim()}
                className="chat-inline-action-button chat-inline-action-send"
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  aria-hidden="true"
                >
                  <path
                    d="M12 19V5"
                    stroke="currentColor"
                    strokeWidth="2.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M6 11L12 5L18 11"
                    stroke="currentColor"
                    strokeWidth="2.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            )}
          </div>
        </form>
        {error ? <p className="chat-input-error">{error}</p> : null}
        <p className="chat-input-note">PRO-MedGraph can make mistakes. Check important information.</p>
      </div>
    </footer>
  );
}

export default ChatInput;