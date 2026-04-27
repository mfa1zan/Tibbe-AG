/* eslint-disable no-console */

const TRACE_ENABLED = String(import.meta.env.VITE_DEBUG_TRACE ?? 'false').toLowerCase() === 'true';

function nowIso() {
  return new Date().toISOString();
}

export function createTraceId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID().slice(0, 8);
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

export function startFrontendTrace({ traceId, query, historyCount, strictMode }) {
  const effectiveTraceId = traceId || createTraceId();
  const startedAt = performance.now();

  if (TRACE_ENABLED) {
    console.groupCollapsed(
      `==================================================\nREQUEST TRACE START\nTrace ID: ${effectiveTraceId}\nTime: ${nowIso()}\n==================================================`
    );
    console.log('[FRONTEND][STEP 1] User submitted message');
    console.table({
      query,
      history_count: historyCount,
      strict_mode: strictMode,
      timestamp: nowIso(),
      trace_id: effectiveTraceId
    });
    console.groupEnd();
  }

  return {
    traceId: effectiveTraceId,
    startedAt
  };
}

export function logFrontendStep(label, payload) {
  if (!TRACE_ENABLED) return;
  console.groupCollapsed(label);
  if (payload && typeof payload === 'object') {
    console.table(payload);
  } else if (payload != null) {
    console.log(payload);
  }
  console.groupEnd();
}

export function endFrontendTrace(trace, payload = {}) {
  if (!trace) return;
  const durationMs = Math.round(performance.now() - trace.startedAt);

  if (TRACE_ENABLED) {
    console.groupCollapsed('[FRONTEND][FINAL STEP] Chat request completed');
    console.table({
      total_request_duration_ms: durationMs,
      trace_id: trace.traceId,
      ...payload
    });
    console.groupEnd();
  }
}

export function isFrontendTraceEnabled() {
  return TRACE_ENABLED;
}
