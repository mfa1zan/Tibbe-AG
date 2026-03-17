import { z } from 'zod';

const chatResponseSchema = z.object({
  final_answer: z.string().min(1),
  evidence_strength: z.string().optional(),
  graph_paths_used: z.number().int().nonnegative().optional(),
  confidence_score: z.number().nullable().optional(),
  safety: z.record(z.unknown()).nullable().optional(),
  reasoning_trace: z.record(z.unknown()).nullable().optional(),
  structured_fields: z.record(z.unknown()).nullable().optional()
});

export const CHAT_API_ERROR_CODE = {
  NETWORK: 'network',
  TIMEOUT: 'timeout',
  CANCELLED: 'cancelled',
  RATE_LIMIT: 'rate_limit',
  SERVER: 'server',
  CLIENT: 'client',
  INVALID_PAYLOAD: 'invalid_payload'
};

export class ChatApiError extends Error {
  constructor(message, { code, status = null, retriable = false, details = null } = {}) {
    super(message);
    this.name = 'ChatApiError';
    this.code = code;
    this.status = status;
    this.retriable = retriable;
    this.details = details;
  }
}

function buildUserMessage(error) {
  if (error instanceof ChatApiError) {
    if (error.code === CHAT_API_ERROR_CODE.CANCELLED) {
      return 'Generation was cancelled.';
    }
    if (error.code === CHAT_API_ERROR_CODE.NETWORK) {
      return 'Unable to connect to backend. Start FastAPI on port 8010 and try again.';
    }
    if (error.code === CHAT_API_ERROR_CODE.TIMEOUT) {
      return 'The request timed out. Please try again.';
    }
    if (error.code === CHAT_API_ERROR_CODE.RATE_LIMIT) {
      return 'The backend is rate-limited right now. Please retry in a moment.';
    }
    if (error.code === CHAT_API_ERROR_CODE.SERVER) {
      return 'Backend returned an internal error. Check FastAPI terminal logs and try again.';
    }
    if (error.code === CHAT_API_ERROR_CODE.INVALID_PAYLOAD) {
      return 'Received an invalid response from backend. Please verify API compatibility.';
    }
    return error.message;
  }

  return 'Failed to send message. Please try again.';
}

export function normalizeChatError(error) {
  const normalized =
    error instanceof ChatApiError
      ? error
      : new ChatApiError('Unexpected chat failure', {
          code: CHAT_API_ERROR_CODE.CLIENT,
          retriable: true,
          details: error
        });

  return {
    code: normalized.code,
    status: normalized.status,
    retriable: normalized.retriable,
    userMessage: buildUserMessage(normalized),
    details: normalized.details
  };
}

function ensureNotAborted(signal) {
  if (signal?.aborted) {
    throw new ChatApiError('Chat request cancelled', {
      code: CHAT_API_ERROR_CODE.CANCELLED,
      retriable: true
    });
  }
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    ensureNotAborted(signal);

    const timeoutId = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);

    const onAbort = () => {
      clearTimeout(timeoutId);
      cleanup();
      reject(
        new ChatApiError('Chat request cancelled', {
          code: CHAT_API_ERROR_CODE.CANCELLED,
          retriable: true
        })
      );
    };

    const cleanup = () => {
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
    };

    if (signal) {
      signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

export async function sendMessageToChatApi(message, options = {}) {
  const { history = [], signal } = options;

  const timeoutController = new AbortController();
  const timeoutId = setTimeout(() => timeoutController.abort(), 30_000);
  const combinedSignal = signal ?? timeoutController.signal;

  let response;
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query: message, history }),
      signal: combinedSignal
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (signal?.aborted) {
        throw new ChatApiError('Chat request cancelled', {
          code: CHAT_API_ERROR_CODE.CANCELLED,
          retriable: true
        });
      }

      throw new ChatApiError('Chat request timed out', {
        code: CHAT_API_ERROR_CODE.TIMEOUT,
        retriable: true
      });
    }

    throw new ChatApiError('Network request failed', {
      code: CHAT_API_ERROR_CODE.NETWORK,
      retriable: true,
      details: error
    });
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    if (response.status === 429) {
      throw new ChatApiError('Rate limit reached', {
        code: CHAT_API_ERROR_CODE.RATE_LIMIT,
        status: response.status,
        retriable: true
      });
    }

    if (response.status >= 500) {
      throw new ChatApiError('Backend internal error', {
        code: CHAT_API_ERROR_CODE.SERVER,
        status: response.status,
        retriable: true
      });
    }

    throw new ChatApiError(`Request failed with status ${response.status}`, {
      code: CHAT_API_ERROR_CODE.CLIENT,
      status: response.status,
      retriable: response.status >= 408
    });
  }

  let rawData;
  try {
    rawData = await response.json();
  } catch (error) {
    throw new ChatApiError('Failed to parse backend JSON response', {
      code: CHAT_API_ERROR_CODE.INVALID_PAYLOAD,
      retriable: false,
      details: error
    });
  }

  const parsedData = chatResponseSchema.safeParse(rawData);
  if (!parsedData.success) {
    throw new ChatApiError('Invalid API response schema', {
      code: CHAT_API_ERROR_CODE.INVALID_PAYLOAD,
      retriable: false,
      details: parsedData.error.flatten()
    });
  }

  const data = parsedData.data;

  return {
    reply: data.final_answer,
    evidenceStrength: typeof data.evidence_strength === 'string' ? data.evidence_strength : 'weak',
    graphPathsUsed: Number.isFinite(data.graph_paths_used) ? data.graph_paths_used : 0,
    confidenceScore: typeof data.confidence_score === 'number' ? data.confidence_score : null,
    safety: data?.safety && typeof data.safety === 'object' ? data.safety : null,
    reasoningTrace: data?.reasoning_trace && typeof data.reasoning_trace === 'object' ? data.reasoning_trace : null,
    structuredFields:
      data?.structured_fields && typeof data.structured_fields === 'object'
        ? data.structured_fields
        : null
  };
}

export async function streamMessageToChatApi(message, options = {}) {
  const {
    onChunk,
    streamChunkSize = 24,
    streamChunkDelayMs = 12,
    ...sendOptions
  } = options;

  const response = await sendMessageToChatApi(message, sendOptions);
  const text = response.reply ?? '';

  if (typeof onChunk !== 'function' || text.length === 0) {
    return response;
  }

  const chunkSize = Math.max(1, streamChunkSize);
  for (let index = 0; index < text.length; index += chunkSize) {
    ensureNotAborted(sendOptions.signal);
    const delta = text.slice(index, index + chunkSize);
    onChunk(delta);
    if (index + chunkSize < text.length) {
      await sleep(streamChunkDelayMs, sendOptions.signal);
    }
  }

  return response;
}