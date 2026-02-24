import { useCallback, useState } from 'react';
import ChatHistory from './components/ChatHistory';
import ChatInput from './components/ChatInput';
import ThemeToggle from './components/ThemeToggle';
import { ThemeProvider } from './context/ThemeContext';
import { sendMessageToChatApi } from './api';

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
    createMessage('bot', 'Hello, I am your biomedical assistant.')
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
      } catch {
        setMessages((current) => current.filter((item) => item.id !== optimisticMessage.id));
        setError('Failed to send message. Please try again.');
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading]
  );

  return (
    <main className="flex h-screen flex-col bg-slate-100 transition-colors dark:bg-slate-950">
      <section className="mx-auto flex h-full w-full max-w-4xl flex-col">
        <header className="border-b border-slate-200 bg-white px-4 py-3 transition-colors dark:border-slate-800 dark:bg-slate-900">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Biomedical Knowledge Chat
            </h1>
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