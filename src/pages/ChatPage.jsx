import { useState } from 'react';
import ChatHistory from '../components/ChatHistory';
import ChatInput from '../components/ChatInput';
import WelcomeScreen from '../components/WelcomeScreen';

function ChatPage({
  messages,
  isLoading,
  hasStreamedToken,
  inputValue,
  error,
  strictMode,
  isWelcomeScreen,
  onInputChange,
  onSendMessage,
  onCancelGeneration
}) {
  const [showStrictModeInfo, setShowStrictModeInfo] = useState(false);

  // Show welcome screen when no active chat
  if (isWelcomeScreen) {
    return (
      <>
        <div className="app-chat-mode-bar" aria-live="polite">
          <span
            className={`app-chat-mode-pill ${strictMode ? 'app-chat-mode-pill-db-only' : 'app-chat-mode-pill-fallback'}`}
          >
            <span>{strictMode ? 'DB-only mode: ON' : 'Model fallback knowledge: ON'}</span>
            <span className="app-chat-mode-info-wrap">
              <button
                type="button"
                className="app-chat-mode-info-button"
                aria-label="What is strict mode?"
                onClick={() => setShowStrictModeInfo((current) => !current)}
              >
                ⓘ
              </button>
              <span className={`app-chat-mode-tooltip ${showStrictModeInfo ? 'app-chat-mode-tooltip-open' : ''}`}>
                Strict mode means answers must come only from Neo4j evidence. If no relevant evidence is found, the
                assistant returns a clear no-data response instead of using general model knowledge.
              </span>
            </span>
          </span>
        </div>

        <WelcomeScreen
          inputValue={inputValue}
          onInputChange={onInputChange}
          onSend={onSendMessage}
          onCancel={onCancelGeneration}
          disabled={isLoading}
          isGenerating={isLoading}
          error={error}
        />
      </>
    );
  }

  return (
    <>
      <div className="app-chat-mode-bar" aria-live="polite">
        <span
          className={`app-chat-mode-pill ${strictMode ? 'app-chat-mode-pill-db-only' : 'app-chat-mode-pill-fallback'}`}
        >
          <span>{strictMode ? 'DB-only mode: ON' : 'Model fallback knowledge: ON'}</span>
          <span className="app-chat-mode-info-wrap">
            <button
              type="button"
              className="app-chat-mode-info-button"
              aria-label="What is strict mode?"
              onClick={() => setShowStrictModeInfo((current) => !current)}
            >
              ⓘ
            </button>
            <span className={`app-chat-mode-tooltip ${showStrictModeInfo ? 'app-chat-mode-tooltip-open' : ''}`}>
              Strict mode means answers must come only from Neo4j evidence. If no relevant evidence is found, the
              assistant returns a clear no-data response instead of using general model knowledge.
            </span>
          </span>
        </span>
      </div>

      <ChatHistory messages={messages} isTyping={isLoading && !hasStreamedToken} />

      <ChatInput
        value={inputValue}
        onChange={onInputChange}
        onSend={onSendMessage}
        onCancel={onCancelGeneration}
        disabled={isLoading}
        isGenerating={isLoading}
        error={error}
      />
    </>
  );
}

export default ChatPage;
