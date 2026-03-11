import { useMemo, useState } from 'react';
import './DebugTracePanel.css';

function DebugTracePanel({ trace }) {
  const [expanded, setExpanded] = useState(false);
  const [copyState, setCopyState] = useState('idle');
  const [copyMinimalState, setCopyMinimalState] = useState('idle');

  const normalized = useMemo(() => {
    const steps = Array.isArray(trace?.steps) ? trace.steps : [];
    const llmCalls = Array.isArray(trace?.llm_calls) ? trace.llm_calls : [];
    const kgQueries = Array.isArray(trace?.kg_queries) ? trace.kg_queries : [];

    const totalSteps = Number.isFinite(trace?.total_steps) ? trace.total_steps : steps.length;
    const totalLlmCalls = Number.isFinite(trace?.total_llm_calls) ? trace.total_llm_calls : llmCalls.length;
    const totalKgQueries = Number.isFinite(trace?.total_kg_queries) ? trace.total_kg_queries : kgQueries.length;
    const totalDurationMs = Number.isFinite(trace?.total_duration_ms) ? trace.total_duration_ms : null;

    return {
      steps,
      llmCalls,
      kgQueries,
      totalSteps,
      totalLlmCalls,
      totalKgQueries,
      totalDurationMs
    };
  }, [trace]);

  if (!trace || typeof trace !== 'object') return null;

  const handleCopyTrace = async () => {
    const orderedText = buildOrderedTraceExport(trace, normalized);
    const copied = await copyTextToClipboard(orderedText);
    setCopyState(copied ? 'success' : 'error');
    setTimeout(() => setCopyState('idle'), 2000);
  };

  const handleCopyMinimalTrace = async () => {
    const minimalText = buildMinimalTraceExport(trace, normalized);
    const copied = await copyTextToClipboard(minimalText);
    setCopyMinimalState(copied ? 'success' : 'error');
    setTimeout(() => setCopyMinimalState('idle'), 2000);
  };

  const copyButtonLabel =
    copyState === 'success'
      ? 'Copied'
      : copyState === 'error'
        ? 'Copy failed'
        : 'Copy Trace';

  const copyMinimalButtonLabel =
    copyMinimalState === 'success'
      ? 'Copied'
      : copyMinimalState === 'error'
        ? 'Copy failed'
        : 'Copy Minimal Trace';

  return (
    <section className="debug-trace-panel" aria-label="Full pipeline debug trace">
      <button
        className="debug-trace-toggle"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        {expanded ? '▾ Hide Full Pipeline Trace' : '▸ View Full Pipeline Trace'}
      </button>

      {expanded ? (
        <div className="debug-trace-body">
          <div className="debug-trace-actions">
            <button
              type="button"
              className="debug-trace-copy-button"
              onClick={handleCopyMinimalTrace}
              aria-label="Copy minimal pipeline trace to clipboard"
            >
              {copyMinimalButtonLabel}
            </button>
            <button
              type="button"
              className="debug-trace-copy-button"
              onClick={handleCopyTrace}
              aria-label="Copy full pipeline trace to clipboard"
            >
              {copyButtonLabel}
            </button>
          </div>

          <h4 className="debug-trace-title">Request Summary</h4>
          <ul className="debug-trace-summary-list">
            <li>Frontend → Backend API calls: 1</li>
            <li>Pipeline steps executed: {normalized.totalSteps}</li>
            <li>LLM API calls: {normalized.totalLlmCalls}</li>
            <li>Neo4j query calls: {normalized.totalKgQueries}</li>
            <li>
              Total pipeline duration:{' '}
              {normalized.totalDurationMs == null ? '—' : `${normalized.totalDurationMs.toFixed(1)} ms`}
            </li>
          </ul>

          <h4 className="debug-trace-title">Step-by-Step Pipeline</h4>
          {normalized.steps.length === 0 ? (
            <p className="debug-trace-empty">No pipeline step records returned.</p>
          ) : (
            <ol className="debug-step-list">
              {normalized.steps.map((step, index) => (
                <li key={`step-${step?.name || index}`} className="debug-step-item">
                  <p className="debug-step-head">
                    <span className="debug-step-index">Step {index + 1}</span>
                    <span className="debug-step-name">{step?.name || 'Unnamed step'}</span>
                    <span className="debug-step-duration">
                      {Number.isFinite(step?.duration_ms) ? `${step.duration_ms.toFixed(1)} ms` : 'duration n/a'}
                    </span>
                  </p>
                  <details className="debug-step-details">
                    <summary>View step input/output</summary>
                    <div className="debug-step-io-grid">
                      <div>
                        <p className="debug-step-io-title">Input</p>
                        <pre>{stringifyForDisplay(step?.input_data)}</pre>
                      </div>
                      <div>
                        <p className="debug-step-io-title">Output</p>
                        <pre>{stringifyForDisplay(step?.output_data)}</pre>
                      </div>
                    </div>
                  </details>
                </li>
              ))}
            </ol>
          )}

          <h4 className="debug-trace-title">LLM Calls</h4>
          {normalized.llmCalls.length === 0 ? (
            <p className="debug-trace-empty">No LLM calls recorded.</p>
          ) : (
            <ol className="debug-call-list">
              {normalized.llmCalls.map((call, index) => (
                <li key={`llm-${index}`} className="debug-call-item">
                  <p>
                    <strong>#{index + 1}</strong> · {call?.purpose || 'unknown purpose'} · model: {call?.model || 'unknown'} ·{' '}
                    {Number.isFinite(call?.duration_ms) ? `${call.duration_ms.toFixed(1)} ms` : 'duration n/a'}
                  </p>
                </li>
              ))}
            </ol>
          )}

          <h4 className="debug-trace-title">Neo4j Queries</h4>
          {normalized.kgQueries.length === 0 ? (
            <p className="debug-trace-empty">No Neo4j queries recorded.</p>
          ) : (
            <ol className="debug-call-list">
              {normalized.kgQueries.map((query, index) => (
                <li key={`kg-${index}`} className="debug-call-item">
                  <p>
                    <strong>#{index + 1}</strong> · {query?.purpose || 'unknown purpose'} · rows:{' '}
                    {Number.isFinite(query?.row_count) ? query.row_count : 'n/a'} ·{' '}
                    {Number.isFinite(query?.duration_ms) ? `${query.duration_ms.toFixed(1)} ms` : 'duration n/a'}
                  </p>
                  <details className="debug-step-details">
                    <summary>View Cypher</summary>
                    <pre>{String(query?.cypher || '')}</pre>
                  </details>
                </li>
              ))}
            </ol>
          )}
        </div>
      ) : null}
    </section>
  );
}

function buildOrderedTraceExport(trace, normalized) {
  const lines = [];
  lines.push('=== PRO-MedGraph Pipeline Debug Trace ===');
  lines.push(`User Input: ${String(trace?.user_input || '')}`);
  lines.push(`Final Output: ${String(trace?.final_output || '')}`);
  lines.push('');

  lines.push('=== Request Summary ===');
  lines.push('Frontend -> Backend API calls: 1');
  lines.push(`Pipeline steps executed: ${normalized.totalSteps}`);
  lines.push(`LLM API calls: ${normalized.totalLlmCalls}`);
  lines.push(`Neo4j query calls: ${normalized.totalKgQueries}`);
  lines.push(
    `Total pipeline duration (ms): ${
      normalized.totalDurationMs == null ? 'n/a' : normalized.totalDurationMs.toFixed(2)
    }`
  );
  lines.push('');

  lines.push('=== Step-by-Step Pipeline ===');
  if (normalized.steps.length === 0) {
    lines.push('No step records.');
  } else {
    normalized.steps.forEach((step, index) => {
      lines.push(
        `[Step ${index + 1}] ${String(step?.name || 'Unnamed step')} | duration_ms=${
          Number.isFinite(step?.duration_ms) ? step.duration_ms.toFixed(2) : 'n/a'
        }`
      );
      lines.push(`INPUT: ${stringifyForExport(step?.input_data)}`);
      lines.push(`OUTPUT: ${stringifyForExport(step?.output_data)}`);
      lines.push('');
    });
  }

  lines.push('=== LLM Calls ===');
  if (normalized.llmCalls.length === 0) {
    lines.push('No LLM calls recorded.');
  } else {
    normalized.llmCalls.forEach((call, index) => {
      lines.push(
        `[#${index + 1}] purpose=${String(call?.purpose || 'unknown')} | model=${String(call?.model || 'unknown')} | duration_ms=${
          Number.isFinite(call?.duration_ms) ? call.duration_ms.toFixed(2) : 'n/a'
        }`
      );
      lines.push(`system_prompt: ${String(call?.system_prompt || '')}`);
      lines.push(`user_prompt: ${String(call?.user_prompt || '')}`);
      lines.push(`response: ${String(call?.response || '')}`);
      lines.push('');
    });
  }

  lines.push('=== Neo4j Queries ===');
  if (normalized.kgQueries.length === 0) {
    lines.push('No Neo4j queries recorded.');
  } else {
    normalized.kgQueries.forEach((query, index) => {
      lines.push(
        `[#${index + 1}] purpose=${String(query?.purpose || 'unknown')} | row_count=${
          Number.isFinite(query?.row_count) ? query.row_count : 'n/a'
        } | duration_ms=${Number.isFinite(query?.duration_ms) ? query.duration_ms.toFixed(2) : 'n/a'}`
      );
      lines.push(`cypher: ${String(query?.cypher || '')}`);
      lines.push(`parameters: ${stringifyForExport(query?.parameters)}`);
      lines.push(`result_sample: ${stringifyForExport(query?.result_sample)}`);
      lines.push('');
    });
  }

  lines.push('=== Raw Trace JSON ===');
  lines.push(stringifyForExport(trace));

  return lines.join('\n');
}

function buildMinimalTraceExport(trace, normalized) {
  const lines = [];
  lines.push('=== PRO-MedGraph Minimal Trace ===');
  lines.push(`User Input: ${String(trace?.user_input || '')}`);
  lines.push(`Final Output: ${String(trace?.final_output || '')}`);
  lines.push(
    `Counts: steps=${normalized.totalSteps}, llm_calls=${normalized.totalLlmCalls}, neo4j_queries=${normalized.totalKgQueries}, duration_ms=${
      normalized.totalDurationMs == null ? 'n/a' : normalized.totalDurationMs.toFixed(2)
    }`
  );
  lines.push('');

  lines.push('--- Steps ---');
  if (normalized.steps.length === 0) {
    lines.push('No steps recorded.');
  } else {
    normalized.steps.forEach((step, index) => {
      lines.push(
        `${index + 1}. ${String(step?.name || 'Unnamed step')} (${Number.isFinite(step?.duration_ms) ? step.duration_ms.toFixed(2) : 'n/a'} ms)`
      );
    });
  }
  lines.push('');

  lines.push('--- LLM Calls ---');
  if (normalized.llmCalls.length === 0) {
    lines.push('No LLM calls recorded.');
  } else {
    normalized.llmCalls.forEach((call, index) => {
      lines.push(
        `${index + 1}. purpose=${String(call?.purpose || 'unknown')} | model=${String(call?.model || 'unknown')} | duration_ms=${Number.isFinite(call?.duration_ms) ? call.duration_ms.toFixed(2) : 'n/a'}`
      );
    });
  }
  lines.push('');

  lines.push('--- Neo4j Queries ---');
  if (normalized.kgQueries.length === 0) {
    lines.push('No Neo4j queries recorded.');
  } else {
    normalized.kgQueries.forEach((query, index) => {
      lines.push(
        `${index + 1}. purpose=${String(query?.purpose || 'unknown')} | rows=${Number.isFinite(query?.row_count) ? query.row_count : 'n/a'} | duration_ms=${Number.isFinite(query?.duration_ms) ? query.duration_ms.toFixed(2) : 'n/a'}`
      );
      if (query?.cypher) {
        lines.push(`   cypher: ${String(query.cypher).replace(/\s+/g, ' ').trim()}`);
      }
    });
  }

  return lines.join('\n');
}

function stringifyForExport(value) {
  if (value == null) return 'null';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

async function copyTextToClipboard(text) {
  if (navigator?.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      return fallbackCopyToClipboard(text);
    }
  }
  return fallbackCopyToClipboard(text);
}

function fallbackCopyToClipboard(text) {
  try {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', 'true');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    document.body.removeChild(textarea);
    return copied;
  } catch {
    return false;
  }
}

function stringifyForDisplay(value) {
  if (value == null) return 'null';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export default DebugTracePanel;
