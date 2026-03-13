import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { CHAT_API_ERROR_CODE, normalizeChatError, streamMessageToChatApi } from './api';
import './App.css';

const ChatPage = lazy(() => import('./pages/ChatPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));

const createMessage = (role, content, metadata = {}) => ({
  id: crypto.randomUUID(),
  role,
  content,
  ...metadata
});

const CHAT_STORAGE_KEY = 'kg-chat-messages-v1';

const INITIAL_GREETING = createMessage('bot', 'Hello, I am PRO-MedGraph. How can I help you today?');

function SidebarToggleIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M10 4.5V19.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path
        d="M7 9.5H17M7 13H13.5M6 5.5H18C19.1 5.5 20 6.4 20 7.5V14.5C20 15.6 19.1 16.5 18 16.5H11L7 19.5V16.5H6C4.9 16.5 4 15.6 4 14.5V7.5C4 6.4 4.9 5.5 6 5.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path
        d="M10 3.8H14L14.6 6C15.1 6.2 15.6 6.4 16 6.7L18.2 5.7L20.3 7.8L19.3 10C19.6 10.4 19.8 10.9 20 11.4L22.2 12V16L20 16.6C19.8 17.1 19.6 17.6 19.3 18L20.3 20.2L18.2 22.3L16 21.3C15.6 21.6 15.1 21.8 14.6 22L14 24.2H10L9.4 22C8.9 21.8 8.4 21.6 8 21.3L5.8 22.3L3.7 20.2L4.7 18C4.4 17.6 4.2 17.1 4 16.6L1.8 16V12L4 11.4C4.2 10.9 4.4 10.4 4.7 10L3.7 7.8L5.8 5.7L8 6.7C8.4 6.4 8.9 6.2 9.4 6L10 3.8Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="14" r="2.4" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function sanitizeStoredMessages(rawValue) {
  if (!Array.isArray(rawValue)) return null;

  const sanitized = rawValue
    .filter((entry) => entry && typeof entry === 'object')
    .map((entry) => ({
      id: typeof entry.id === 'string' && entry.id ? entry.id : crypto.randomUUID(),
      role: entry.role === 'user' ? 'user' : 'bot',
      content: typeof entry.content === 'string' ? entry.content : '',
      variant: entry.variant === 'status' ? 'status' : undefined,
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
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
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

  const handleClearConversation = useCallback(() => {
    requestAbortRef.current?.abort();
    setMessages([INITIAL_GREETING]);
    setIsLoading(false);
    setHasStreamedToken(false);
    setError('');

    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch (storageError) {
      void storageError;
    }
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
        setMessages((current) => {
          if (normalizedError.code === CHAT_API_ERROR_CODE.CANCELLED) {
            const streamingMessage = current.find((item) => item.id === streamingBotMessageId);
            const partialContent =
              streamingMessage && typeof streamingMessage.content === 'string'
                ? streamingMessage.content.trim()
                : '';

            const withoutStreaming = current.filter((item) => item.id !== streamingBotMessageId);

            if (partialContent) {
              return [
                ...withoutStreaming,
                createMessage('bot', partialContent),
                createMessage('bot', 'Request aborted by user.', { variant: 'status' })
              ];
            }

            return [...withoutStreaming, createMessage('bot', 'Request aborted by user.', { variant: 'status' })];
          }

          return current.filter((item) => item.id !== optimisticMessage.id && item.id !== streamingBotMessageId);
        });
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
      <aside
        className={`app-sidebar ${isSidebarExpanded ? 'app-sidebar-expanded' : 'app-sidebar-collapsed'}`}
        aria-label="Primary navigation"
      >
        <div className="app-sidebar-top">
          <button
            type="button"
            className="app-sidebar-logo-button"
            aria-label={isSidebarExpanded ? 'Collapse sidebar' : 'Open sidebar'}
            title={isSidebarExpanded ? 'Collapse sidebar' : 'Open sidebar'}
            onClick={() => setIsSidebarExpanded((current) => !current)}
          >
            <SidebarToggleIcon />
          </button>

          {isSidebarExpanded ? (
            <>
              <div className="app-sidebar-brand-text">
                <h1 className="app-title">PRO-MedGraph</h1>
                <p className="app-sidebar-subtitle">Biomedical assistant</p>
              </div>
            </>
          ) : null}
        </div>

        <nav className="app-nav" aria-label="Primary">
          <NavLink
            to="/chat"
            className={({ isActive }) => `app-nav-link ${isActive ? 'app-nav-link-active' : ''}`}
            title="Chat"
          >
            <span className="app-nav-icon" aria-hidden="true"><ChatIcon /></span>
            {isSidebarExpanded ? <span className="app-nav-label">Chat</span> : null}
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) => `app-nav-link ${isActive ? 'app-nav-link-active' : ''}`}
            title="Settings"
          >
            <span className="app-nav-icon" aria-hidden="true"><SettingsIcon /></span>
            {isSidebarExpanded ? <span className="app-nav-label">Settings</span> : null}
          </NavLink>
        </nav>
      </aside>

      <section className="app-main">
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route
              path="/"
              element={<Navigate to="/chat" replace />}
            />
            <Route
              path="/chat"
              element={
                <ChatPage
                  messages={messages}
                  isLoading={isLoading}
                  hasStreamedToken={hasStreamedToken}
                  inputValue={inputValue}
                  error={error}
                  onInputChange={setInputValue}
                  onSendMessage={handleSendMessage}
                  onCancelGeneration={handleCancelGeneration}
                />
              }
            />
            <Route
              path="/settings"
              element={
                <SettingsPage
                  messages={messages}
                  onClearConversation={handleClearConversation}
                />
              }
            />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </Suspense>
      </section>
    </main>
  );
}

function RouteFallback() {
  return (
    <section className="app-route-panel">
      <p className="app-route-helper">Loading page...</p>
    </section>
  );
}

export default App;