import ChatInput from './ChatInput';
import './WelcomeScreen.css';

function WelcomeScreen({ inputValue, onInputChange, onSend, onCancel, disabled, isGenerating, error }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"
              fill="url(#welcome-gradient)"
              opacity="0.15"
            />
            <path
              d="M7 9.5H17M7 13H13.5M6 5.5H18C19.1 5.5 20 6.4 20 7.5V14.5C20 15.6 19.1 16.5 18 16.5H11L7 19.5V16.5H6C4.9 16.5 4 15.6 4 14.5V7.5C4 6.4 4.9 5.5 6 5.5Z"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <defs>
              <linearGradient id="welcome-gradient" x1="2" y1="2" x2="22" y2="22">
                <stop stopColor="var(--primary-color)" />
                <stop offset="1" stopColor="var(--primary-color)" stopOpacity="0.4" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <h2 className="welcome-heading">What can I help you with?</h2>
        <p className="welcome-subtext">
          Ask any biomedical question &mdash; powered by Knowledge Graph + AI
        </p>
      </div>

      <div className="welcome-input-wrapper">
        <ChatInput
          value={inputValue}
          onChange={onInputChange}
          onSend={onSend}
          onCancel={onCancel}
          disabled={disabled}
          isGenerating={isGenerating}
          error={error}
        />
      </div>
    </div>
  );
}

export default WelcomeScreen;
