import { useCallback, useEffect, useRef, useState } from 'react';
import ChatHistory from './components/ChatHistory';
import ChatInput from './components/ChatInput';
import ThemeToggle from './components/ThemeToggle';
import { ThemeProvider } from './context/ThemeContext';
import { normalizeChatError, sendMessageToChatApi } from './api';
import './App.css';

const createMessage = (role, content, metadata = {}) => ({
  id: crypto.randomUUID(),
  role,
  content,
  ...metadata
});

function App() {
  return (
    <ThemeProvider>
      <AppShell />
    </ThemeProvider>
  );
}

function AppShell() {
  const [messages, setMessages] = useState([
    createMessage('bot', 'Hello, I am PRO-MedGraph. How can I help you today?')
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [inputValue, setInputValue] = useState('');
  const messagesRef = useRef(messages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const handleSendMessage = useCallback(
    async (messageText) => {
      if (isLoading || !messageText.trim()) {
        return;
      }

      setError('');
      setIsLoading(true);

      // Clear input immediately so the field is empty and ready for next message
      setInputValue('');

      const optimisticMessage = createMessage('user', messageText.trim());
      setMessages((current) => [...current, optimisticMessage]);

      try {
        // Build conversation history for co-reference resolution (last 10 messages)
        const historyForApi = [...messagesRef.current, optimisticMessage]
          .slice(-10)
          .map((m) => ({ role: m.role === 'bot' ? 'bot' : 'user', content: m.content }));

        const { reply, evidenceStrength, graphPathsUsed, confidenceScore, safety, reasoningTrace } =
          await sendMessageToChatApi(messageText, {
            usePlaceholder: import.meta.env.VITE_USE_PLACEHOLDER_BOT === 'true',
            history: historyForApi
          });

        setMessages((current) => [
          ...current,
          createMessage('bot', reply, {
            evidenceStrength,
            graphPathsUsed,
            confidenceScore,
            safety,
            reasoningTrace
          })
        ]);
      } catch (error) {
        const normalizedError = normalizeChatError(error);
        setMessages((current) => current.filter((item) => item.id !== optimisticMessage.id));
        setError(normalizedError.userMessage);
      } finally {
        setIsLoading(false);
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

        <ChatHistory messages={messages} isTyping={isLoading} />

        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          disabled={isLoading}
          error={error}
        />
      </section>
    </main>
  );
}

export default App;