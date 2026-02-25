const PLACEHOLDER_BOT_REPLY = 'Hello, I am PRO-MedGraph. How can I help you today?';

export async function sendMessageToChatApi(message, options = {}) {
  const { usePlaceholder = false } = options;

  if (usePlaceholder) {
    return {
      reply: PLACEHOLDER_BOT_REPLY,
      evidenceStrength: 'moderate',
      graphPathsUsed: 1,
      confidenceScore: 0.6,
      safety: null
    };
  }

  let response;
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query: message })
    });
  } catch {
    throw new Error('Unable to connect to backend. Start FastAPI on port 8010 and try again.');
  }

  if (!response.ok) {
    if (response.status >= 500) {
      throw new Error('Backend returned 500. Check FastAPI terminal logs and ensure env values are valid.');
    }

    throw new Error(`Request failed with status ${response.status}`);
  }

  const data = await response.json();

  if (!data?.final_answer || typeof data.final_answer !== 'string') {
    throw new Error('Invalid API response: missing final_answer');
  }

  return {
    reply: data.final_answer,
    evidenceStrength: typeof data.evidence_strength === 'string' ? data.evidence_strength : 'weak',
    graphPathsUsed: Number.isFinite(data.graph_paths_used) ? data.graph_paths_used : 0,
    confidenceScore: typeof data.confidence_score === 'number' ? data.confidence_score : null,
    safety: data?.safety && typeof data.safety === 'object' ? data.safety : null
  };
}