const PLACEHOLDER_BOT_REPLY = 'Hello, I am your biomedical assistant.';

export async function sendMessageToChatApi(message, options = {}) {
  const { usePlaceholder = false } = options;

  if (usePlaceholder) {
    return PLACEHOLDER_BOT_REPLY;
  }

  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message })
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  const data = await response.json();

  if (!data?.reply || typeof data.reply !== 'string') {
    throw new Error('Invalid API response: missing reply');
  }

  return data.reply;
}