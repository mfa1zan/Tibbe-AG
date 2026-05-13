import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import { CHAT_API_ERROR_CODE, normalizeChatError, streamMessageToChatApi, listSessions, createSession, deleteSession, deleteEmptySessions, getSessionMessages, updateSession } from './api';
import './App.css';

const ChatPage = lazy(() => import('./pages/ChatPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));

const createMessage = (role, content, metadata = {}) => ({
  id: crypto.randomUUID(),
  role,
  content,
  ...metadata
});

/* ─── localStorage keys ─── */
const SESSIONS_STORAGE_KEY = 'kg-chat-sessions-v1';
const STRICT_MODE_STORAGE_KEY = 'kg-chat-strict-mode-v1';

function chatMessagesKey(chatId) {
  return `kg-chat-msg-${chatId}`;
}

/* ─── SVG Icons ─── */

function SidebarToggleIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M10 4.5V19.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function NewChatIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path
        d="M12 5V19M5 12H19"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
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

function DeleteIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" width="14" height="14">
      <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MoreActionsIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" width="16" height="16">
      <circle cx="6" cy="12" r="1.8" fill="currentColor" />
      <circle cx="12" cy="12" r="1.8" fill="currentColor" />
      <circle cx="18" cy="12" r="1.8" fill="currentColor" />
    </svg>
  );
}

function ProfileIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" width="20" height="20">
      <circle cx="12" cy="8" r="3.5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M4 20C4 16.6863 7.13401 14 11 14H13C16.866 14 20 16.6863 20 20" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

/* ─── Helpers ─── */

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

async function loadSessions() {
  try {
    const sessions = await listSessions();
    return sessions.map(session => ({
      id: session.id,
      title: session.title,
      createdAt: session.created_at,
      updatedAt: session.updated_at,
    }));
  } catch (error) {
    console.error('Failed to load sessions:', error);
    return [];
  }
}

async function saveSessions(sessions) {
  // Sessions are now managed by the backend, no need to save locally
  // This function is kept for compatibility but does nothing
}

async function loadMessagesForChat(chatId) {
  try {
    const messages = await getSessionMessages(chatId);
    return sanitizeStoredMessages(messages);
  } catch (error) {
    console.error('Failed to load messages for chat:', error);
    return [];
  }
}

function saveMessagesForChat(chatId, messages) {
  // Messages are now saved automatically by the chat API
  // This function is kept for compatibility but does nothing
}

async function deleteChat(chatId) {
  try {
    await deleteSession(chatId);
  } catch (error) {
    console.error('Failed to delete chat:', error);
  }
}

function loadInitialStrictMode() {
  try {
    return localStorage.getItem(STRICT_MODE_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

async function getGravatarUrl(email) {
  const normalized = email.trim().toLowerCase();
  const encoder = new TextEncoder();
  const data = encoder.encode(normalized);
  const hashBuffer = await crypto.subtle.digest('MD5', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hash = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  return `https://www.gravatar.com/avatar/${hash}?d=identicon&s=128`;
}

/** Derive a short title from the first user message. */
function deriveChatTitle(messageText) {
  const cleaned = messageText.trim().replace(/\n+/g, ' ');
  if (cleaned.length <= 40) return cleaned;
  return cleaned.slice(0, 40).trimEnd() + '…';
}

/* ─── App ─── */

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </ThemeProvider>
  );
}

function AppShell() {
  const { user, loading, logout } = useAuth();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null); // null = welcome screen
  const [messages, setMessages] = useState([]);
  const [strictMode, setStrictMode] = useState(loadInitialStrictMode);
  const [isLoading, setIsLoading] = useState(false);
  const [hasStreamedToken, setHasStreamedToken] = useState(false);
  const [error, setError] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const [openActionsMenuChatId, setOpenActionsMenuChatId] = useState(null);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const [profileImageUrl, setProfileImageUrl] = useState('');
  const messagesRef = useRef(messages);
  const requestAbortRef = useRef(null);

  // Load sessions when user changes
  useEffect(() => {
    if (user) {
      setActiveChatId(null);
      setMessages([]);
      setInputValue('');
      setError('');
      setIsLoading(false);
      setHasStreamedToken(false);
      
      // Clean up any empty sessions and then load the remaining sessions
      deleteEmptySessions()
        .catch((error) => console.error('Failed to clean up empty sessions:', error))
        .finally(() => {
          loadSessions().then(setSessions).catch(console.error);
        });
    } else {
      setSessions([]);
      setActiveChatId(null);
      setMessages([]);
      setInputValue('');
      setError('');
      setIsLoading(false);
      setHasStreamedToken(false);
    }
  }, [user]);

  // Keep messagesRef in sync
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Persist messages whenever they change (and we have an active chat)
  useEffect(() => {
    if (activeChatId) {
      saveMessagesForChat(activeChatId, messages);
    }
  }, [messages, activeChatId]);

  // Persist strict mode
  useEffect(() => {
    try {
      localStorage.setItem(STRICT_MODE_STORAGE_KEY, String(strictMode));
    } catch {
      /* noop */
    }
  }, [strictMode]);

  // Cleanup abort on unmount
  useEffect(() => () => requestAbortRef.current?.abort(), []);

  useEffect(() => {
    let mounted = true;
    if (!user?.email) {
      setProfileImageUrl('');
      return;
    }

    getGravatarUrl(user.email).then((url) => {
      if (mounted) {
        setProfileImageUrl(url);
      }
    });

    return () => {
      mounted = false;
    };
  }, [user?.email]);

  useEffect(() => {
    if (!isProfileMenuOpen) return undefined;

    const handleDocumentClick = () => setIsProfileMenuOpen(false);
    document.addEventListener('click', handleDocumentClick);
    return () => document.removeEventListener('click', handleDocumentClick);
  }, [isProfileMenuOpen]);

  // Sorted sessions (most recent first)
  const sortedSessions = useMemo(
    () => [...sessions].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)),
    [sessions]
  );

  /* ─── Handlers ─── */

  const handleNewChat = useCallback(async () => {
    requestAbortRef.current?.abort();
    setError('');
    // Don't create a session immediately - just prepare an empty chat
    // The session will be created when the first message is sent
    setActiveChatId(null); // Use a temporary ID to indicate a new unsaved chat
    setMessages([]);
    setIsLoading(false);
    setHasStreamedToken(false);
    setInputValue('');
    navigate('/chat');
  }, [navigate]);

  const handleLogout = useCallback(async () => {
    // Clean up any empty sessions before logout
    try {
      await deleteEmptySessions();
    } catch (error) {
      console.error('Failed to clean up empty sessions:', error);
      // Continue with logout even if cleanup fails
    }
    logout();
    setIsProfileMenuOpen(false);
    navigate('/login');
  }, [logout, navigate]);

  const toggleProfileMenu = useCallback((event) => {
    event.stopPropagation();
    setIsProfileMenuOpen((current) => !current);
  }, []);

  const handleSelectChat = useCallback(async (chatId) => {
    requestAbortRef.current?.abort();
    setOpenActionsMenuChatId(null);
    setActiveChatId(chatId);
    try {
      const chatMessages = await loadMessagesForChat(chatId);
      setMessages(chatMessages);
    } catch (error) {
      console.error('Failed to load chat messages:', error);
      setMessages([]);
    }
    setIsLoading(false);
    setHasStreamedToken(false);
    setError('');
    setInputValue('');
    navigate('/chat');
  }, [navigate]);

  const handleDeleteChat = useCallback(
    async (chatId, event) => {
      event.stopPropagation();
      setOpenActionsMenuChatId(null);
      try {
        await deleteChat(chatId);
        setSessions((prev) => prev.filter((s) => s.id !== chatId));
        // If we just deleted the active chat, go to welcome screen
        if (activeChatId === chatId) {
          setActiveChatId(null);
          setMessages([]);
          setIsLoading(false);
          setHasStreamedToken(false);
          setError('');
        }
      } catch (error) {
        console.error('Failed to delete chat:', error);
      }
    },
    [activeChatId]
  );

  const handleRenameChat = useCallback((chatId, currentTitle, event) => {
    event.stopPropagation();

    const nextTitle = window.prompt('Rename chat', currentTitle);
    if (nextTitle === null) {
      return;
    }

    const trimmed = nextTitle.trim();
    if (!trimmed) {
      return;
    }

    setOpenActionsMenuChatId(null);
    setSessions((prev) => {
      const updated = prev.map((session) =>
        session.id === chatId ? { ...session, title: trimmed, updatedAt: Date.now() } : session
      );
      saveSessions(updated);
      return updated;
    });
  }, []);

  const handleCancelGeneration = useCallback(() => {
    requestAbortRef.current?.abort();
  }, []);

  const handleSendMessage = useCallback(
    async (messageText) => {
      if (isLoading || !messageText.trim()) return;

      setError('');
      setIsLoading(true);
      setHasStreamedToken(false);
      setInputValue('');

      const optimisticMessage = createMessage('user', messageText.trim());
      const streamingBotMessageId = crypto.randomUUID();
      const abortController = new AbortController();
      requestAbortRef.current = abortController;

      // If no active chat, create a new session
      let currentChatId = activeChatId;
      if (!currentChatId) {
        try {
          const newSession = await createSession(deriveChatTitle(messageText));
          currentChatId = newSession.id;
          setActiveChatId(currentChatId);
          setSessions((prev) => [newSession, ...prev]);
          navigate('/chat');
        } catch (error) {
          console.error('Failed to create new chat session:', error);
          setError('Failed to create new chat session');
          setIsLoading(false);
          return;
        }
      } else {
        // Session will be updated automatically by the backend when messages are saved
      }

      setMessages((current) => [
        ...current,
        optimisticMessage,
        createMessage('bot', '', { isStreaming: true, id: streamingBotMessageId })
      ]);

      try {
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
        } = await streamMessageToChatApi(messageText, {
          history: historyForApi,
          strictMode,
          sessionId: currentChatId,
          signal: abortController.signal,
          onChunk: (chunk) => {
            setHasStreamedToken(true);
            setMessages((current) =>
              current.map((item) =>
                item.id === streamingBotMessageId
                  ? { ...item, content: `${item.content}${chunk}` }
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

        try {
          const newTitle = deriveChatTitle(messageText);
          const updatedSession = await updateSession(currentChatId, { title: newTitle });
          setSessions((prev) => prev.map((s) =>
            s.id === currentChatId ? { ...s, title: updatedSession.title, updatedAt: updatedSession.updated_at } : s
          ));
        } catch (error) {
          console.error('Failed to update session title:', error);
        }
      } catch (err) {
        const normalizedError = normalizeChatError(err);
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
            return [
              ...withoutStreaming,
              createMessage('bot', 'Request aborted by user.', { variant: 'status' })
            ];
          }
          return current.filter(
            (item) => item.id !== optimisticMessage.id && item.id !== streamingBotMessageId
          );
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
    [isLoading, strictMode, activeChatId]
  );

  const isWelcomeScreen = activeChatId === null;

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner">Loading...</div>
      </div>
    );
  }

  return (
    <main className={`app-shell ${!user ? 'app-shell-auth' : ''}`}>
      {user && (
        <aside
          className={`app-sidebar ${isSidebarExpanded ? 'app-sidebar-expanded' : 'app-sidebar-collapsed'}`}
          aria-label="Primary navigation"
        >
        {/* ── Top: logo + brand ── */}
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
            <div className="app-sidebar-brand-text">
              <h1 className="app-title">PRO-MedGraph</h1>
              <p className="app-sidebar-subtitle">Biomedical assistant</p>
            </div>
          ) : null}
        </div>

        {/* ── New Chat button ── */}
        <button
          type="button"
          className="app-new-chat-button"
          title="New Chat"
          onClick={handleNewChat}
        >
          <span className="app-nav-icon" aria-hidden="true"><NewChatIcon /></span>
          {isSidebarExpanded ? <span className="app-nav-label">New Chat</span> : null}
        </button>

        {/* ── Chat history list ── */}
        {isSidebarExpanded && sortedSessions.length > 0 ? (
          <div className="app-sidebar-history">
            <p className="app-sidebar-history-label">Recent Chats</p>
            <ul className="app-sidebar-history-list">
              {sortedSessions.map((session) => (
                <li key={session.id}>
                  <button
                    type="button"
                    className={`app-sidebar-history-item ${session.id === activeChatId ? 'app-sidebar-history-item-active' : ''}`}
                    onClick={() => handleSelectChat(session.id)}
                    title={session.title}
                  >
                    <span className="app-sidebar-history-icon"><ChatIcon /></span>
                    <span className="app-sidebar-history-title">{session.title}</span>
                    <span
                      className="app-sidebar-history-actions"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <button
                        type="button"
                        className="app-sidebar-history-actions-toggle"
                        aria-haspopup="menu"
                        aria-expanded={openActionsMenuChatId === session.id}
                        aria-label={`Open actions for chat: ${session.title}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setOpenActionsMenuChatId((current) => (current === session.id ? null : session.id));
                        }}
                      >
                        <MoreActionsIcon />
                      </button>

                      {openActionsMenuChatId === session.id ? (
                        <span className="app-sidebar-history-actions-menu" role="menu">
                          <button
                            type="button"
                            className="app-sidebar-history-menu-item"
                            role="menuitem"
                            onClick={(event) => handleRenameChat(session.id, session.title, event)}
                          >
                            Rename
                          </button>
                          <button
                            type="button"
                            className="app-sidebar-history-menu-item app-sidebar-history-menu-item-delete"
                            role="menuitem"
                            onClick={(event) => handleDeleteChat(session.id, event)}
                          >
                            <DeleteIcon />
                            Delete
                          </button>
                        </span>
                      ) : null}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {/* ── Settings at the bottom ── */}
        <nav className="app-sidebar-bottom" aria-label="User navigation">
          <div className="app-nav">
            <NavLink
              to="/settings"
              className={({ isActive }) => `app-nav-link ${isActive ? 'app-nav-link-active' : ''}`}
              title="Settings"
            >
              <span className="app-nav-icon" aria-hidden="true"><SettingsIcon /></span>
              {isSidebarExpanded ? <span className="app-nav-label">Settings</span> : null}
            </NavLink>

            <div className="app-profile-bottom-group" onClick={(event) => event.stopPropagation()}>
              <button
                type="button"
                className={`app-nav-link app-profile-bottom-button ${isProfileMenuOpen ? 'app-nav-link-active' : ''}`}
                aria-haspopup="menu"
                aria-expanded={isProfileMenuOpen}
                onClick={toggleProfileMenu}
                title="Open profile menu"
              >
                <span className="app-nav-icon" aria-hidden="true">
                  {profileImageUrl ? (
                    <img className="app-profile-avatar" src={profileImageUrl} alt="Profile avatar" />
                  ) : (
                    <ProfileIcon />
                  )}
                </span>
                {isSidebarExpanded ? <span className="app-nav-label">{user?.display_name || 'Profile'}</span> : null}
              </button>

              {isProfileMenuOpen ? (
                <div className="app-profile-menu app-profile-menu-bottom" role="menu" aria-label="User profile menu">
                  <div className="app-profile-menu-item">
                    <span>Email</span>
                    <strong>{user?.email}</strong>
                  </div>
                  <button type="button" className="app-profile-menu-logout" onClick={handleLogout}>
                    Logout
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </nav>
      </aside>
      )}

      <section className="app-main">
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route
              path="/"
              element={<Navigate to={user ? "/chat" : "/login"} replace />}
            />
            <Route
              path="/login"
              element={user ? <Navigate to="/chat" replace /> : <LoginPage />}
            />
            <Route
              path="/register"
              element={user ? <Navigate to="/chat" replace /> : <RegisterPage />}
            />
            <Route
              path="/chat"
              element={user ? (
                <ChatPage
                  messages={messages}
                  isLoading={isLoading}
                  hasStreamedToken={hasStreamedToken}
                  inputValue={inputValue}
                  error={error}
                  strictMode={strictMode}
                  isWelcomeScreen={isWelcomeScreen}
                  onInputChange={setInputValue}
                  onSendMessage={handleSendMessage}
                  onCancelGeneration={handleCancelGeneration}
                />
              ) : <Navigate to="/login" replace />}
            />
            <Route
              path="/settings"
              element={user ? (
                <SettingsPage
                  strictMode={strictMode}
                  onStrictModeChange={setStrictMode}
                />
              ) : <Navigate to="/login" replace />}
            />
            <Route path="*" element={<Navigate to={user ? "/chat" : "/login"} replace />} />
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