import { useCallback, useState } from 'react';
import ChatHistory from './components/ChatHistory';
import ChatInput from './components/ChatInput';
import { sendMessageToChatApi } from './api';

const createMessage = (role, content) => ({
  id: crypto.randomUUID(),
  role,
  content
});

const BOT_REPLY_DELAY_MS = 600;

function App() {
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
        const reply = await sendMessageToChatApi(messageText, {
          usePlaceholder: import.meta.env.VITE_USE_PLACEHOLDER_BOT === 'true'
        });

        await new Promise((resolve) => {
          window.setTimeout(resolve, BOT_REPLY_DELAY_MS);
        });

        setMessages((current) => [...current, createMessage('bot', reply)]);
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
    <main className="flex h-screen flex-col bg-slate-100">
      <section className="mx-auto flex h-full w-full max-w-4xl flex-col">
        <header className="border-b border-slate-200 bg-white px-4 py-3">
          <h1 className="text-lg font-semibold text-slate-900">Biomedical Knowledge Chat</h1>
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