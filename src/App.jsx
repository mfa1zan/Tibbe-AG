import { useCallback, useEffect, useRef, useState } from 'react';
import ChatHistory from './components/ChatHistory';
import ChatInput from './components/ChatInput';
import ThemeToggle from './components/ThemeToggle';
import { ThemeProvider } from './context/ThemeContext';
import { CHAT_API_ERROR_CODE, normalizeChatError, streamMessageToChatApi } from './api';
import './App.css';

const createMessage = (role, content, metadata = {}) => ({
  id: crypto.randomUUID(),
  role,
  content,
  ...metadata
});

const CHAT_STORAGE_KEY = 'kg-chat-messages-v1';

const INITIAL_GREETING = createMessage('bot', 'Hello, I am PRO-MedGraph. How can I help you today?');

function sanitizeStoredMessages(rawValue) {
  if (!Array.isArray(rawValue)) return null;

  const sanitized = rawValue
    .filter((entry) => entry && typeof entry === 'object')
    .map((entry) => ({
      id: typeof entry.id === 'string' && entry.id ? entry.id : crypto.randomUUID(),
      role: entry.role === 'user' ? 'user' : 'bot',
      content: typeof entry.content === 'string' ? entry.content : '',
      confidenceScore: typeof entry.confidenceScore === 'number' ? entry.confidenceScore : null,
      evidenceStrength: typeof entry.evidenceStrength === 'string' ? entry.evidenceStrength : undefined,
      graphPathsUsed: Number.isFinite(entry.graphPathsUsed) ? entry.graphPathsUsed : undefined,
      reasoningTrace:
        entry.reasoningTrace && typeof entry.reasoningTrace === 'object' ? entry.reasoningTrace : null,
      safety: entry.safety && typeof entry.safety === 'object' ? entry.safety : null,
      structuredFields:
        entry.structuredFields && typeof entry.structuredFields === 'object' ? entry.structuredFields : null
    }))
    .filter((entry) => entry.content.trim().length > 0);

  return sanitized.length > 0 ? sanitized : null;
}

function loadInitialMessages() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return [INITIAL_GREETING];
    const parsed = JSON.parse(raw);
    return sanitizeStoredMessages(parsed) ?? [INITIAL_GREETING];
  } catch {
    return [INITIAL_GREETING];
  }
}

function App() {
  return (
    <ThemeProvider>
      <AppShell />
    </ThemeProvider>
  );
}

function AppShell() {
  const [messages, setMessages] = useState(loadInitialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [hasStreamedToken, setHasStreamedToken] = useState(false);
  const [error, setError] = useState('');
  const [inputValue, setInputValue] = useState('');
  const messagesRef = useRef(messages);
  const requestAbortRef = useRef(null);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    try {
      const persistableMessages = messages.filter((message) => !message.isStreaming);
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(persistableMessages));
    } catch (storageError) {
      void storageError;
    }
  }, [messages]);

  useEffect(() => () => requestAbortRef.current?.abort(), []);

  const handleCancelGeneration = useCallback(() => {
    requestAbortRef.current?.abort();
  }, []);

  const handleSendMessage = useCallback(
    async (messageText) => {
      if (isLoading || !messageText.trim()) {
        return;
      }

      setError('');
      setIsLoading(true);
      setHasStreamedToken(false);

      // Clear input immediately so the field is empty and ready for next message
      setInputValue('');

      const optimisticMessage = createMessage('user', messageText.trim());
      const streamingBotMessageId = crypto.randomUUID();
      const abortController = new AbortController();
      requestAbortRef.current = abortController;

      setMessages((current) => [
        ...current,
        optimisticMessage,
        createMessage('bot', '', { isStreaming: true, id: streamingBotMessageId })
      ]);

      try {
        // Build conversation history for co-reference resolution (last 10 messages)
        const historyForApi = [...messagesRef.current, optimisticMessage]
          .slice(-10)
          .map((m) => ({ role: m.role === 'bot' ? 'bot' : 'user', content: m.content }));

        const {
          reply,
          evidenceStrength,
          graphPathsUsed,
          confidenceScore,
          safety,
          reasoningTrace,
          structuredFields
        } =
          await streamMessageToChatApi(messageText, {
            usePlaceholder: import.meta.env.VITE_USE_PLACEHOLDER_BOT === 'true',
            history: historyForApi,
            signal: abortController.signal,
            onChunk: (chunk) => {
              setHasStreamedToken(true);
              setMessages((current) =>
                current.map((item) =>
                  item.id === streamingBotMessageId
                    ? {
                        ...item,
                        content: `${item.content}${chunk}`
                      }
                    : item
                )
              );
            }
          });

        setMessages((current) =>
          current.map((item) =>
            item.id === streamingBotMessageId
              ? {
                  ...item,
                  content: reply,
                  isStreaming: false,
                  evidenceStrength,
                  graphPathsUsed,
                  confidenceScore,
                  safety,
                  reasoningTrace,
                  structuredFields
                }
              : item
          )
        );
      } catch (error) {
        const normalizedError = normalizeChatError(error);
        setMessages((current) =>
          current.filter((item) => {
            if (normalizedError.code === CHAT_API_ERROR_CODE.CANCELLED) {
              return item.id !== streamingBotMessageId;
            }

            return item.id !== optimisticMessage.id && item.id !== streamingBotMessageId;
          })
        );
        if (normalizedError.code !== CHAT_API_ERROR_CODE.CANCELLED) {
          setError(normalizedError.userMessage);
        }
      } finally {
        requestAbortRef.current = null;
        setIsLoading(false);
        setHasStreamedToken(false);
      }
    },
    [isLoading]
  );

  return (
    <main className="app-shell">
      <section className="app-container">
        <header className="app-header">
          <div className="app-header-row">
            <h1 className="app-title">PRO-MedGraph</h1>
            <ThemeToggle />
          </div>
        </header>

        <ChatHistory messages={messages} isTyping={isLoading && !hasStreamedToken} />

        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onCancel={handleCancelGeneration}
          disabled={isLoading}
          isGenerating={isLoading}
          error={error}
        />
      </section>
    </main>
  );
}

export default App;