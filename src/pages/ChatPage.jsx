import ChatHistory from '../components/ChatHistory';
import ChatInput from '../components/ChatInput';

function ChatPage({
  messages,
  isLoading,
  hasStreamedToken,
  inputValue,
  error,
  onInputChange,
  onSendMessage,
  onCancelGeneration
}) {
  return (
    <>
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
