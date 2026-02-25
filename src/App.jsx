import { useCallback, useState } from 'react';
import ChatHistory from './components/ChatHistory';
import ChatInput from './components/ChatInput';
import ThemeToggle from './components/ThemeToggle';
import { ThemeProvider } from './context/ThemeContext';
import { sendMessageToChatApi } from './api';
import './App.css';

const createMessage = (role, content, metadata = {}) => ({
  id: crypto.randomUUID(),
  role,
  content,
  ...metadata
});

const SESSION_ID = crypto.randomUUID();

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

  const handleSendMessage = useCallback(
    async (messageText) => {
      if (isLoading || !messageText.trim()) {
        return;
      }

      setError('');
      setIsLoading(true);

      const optimisticMessage = createMessage('user', messageText.trim());
      setMessages((current) => [...current, optimisticMessage]);

      try {
        const { reply, provenance } = await sendMessageToChatApi(messageText, {
          sessionId: SESSION_ID,
          usePlaceholder: import.meta.env.VITE_USE_PLACEHOLDER_BOT === 'true'
        });

        setMessages((current) => [
          ...current,
          createMessage('bot', reply, {
            provenance
          })
        ]);
        setInputValue('');
      } catch (error) {
        setMessages((current) => current.filter((item) => item.id !== optimisticMessage.id));
        setError(error instanceof Error ? error.message : 'Failed to send message. Please try again.');
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