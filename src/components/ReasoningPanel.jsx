import { useState } from 'react';
import './ReasoningPanel.css';

/**
 * Collapsible panel showing the full reasoning trace from the
 * research-grade GraphRAG pipeline.
 */
function ReasoningPanel({ trace }) {
  const [expanded, setExpanded] = useState(false);

  if (!trace) return null;

  const {
    entity_detected,
    rationale_plan,
    causal_ranking,
    causal_summary,
    dosage_validation,
    faith_alignment_notes,
    faith_alignment_score,
    confidence_breakdown,
    multi_hop_activated,
    evaluation_metrics,
    pipeline_stages
  } = trace;

  return (
    <div className="reasoning-panel">
      <button
        className="reasoning-toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? '▾ Hide Reasoning Trace' : '▸ View Reasoning Trace'}
      </button>

      {expanded && (
        <div className="reasoning-body">
          {/* Entities detected */}
          {entity_detected && (
            <Section title="Entities Detected">
              <KeyValue label="Disease" value={entity_detected.disease} />
              <KeyValue label="Ingredient" value={entity_detected.ingredient} />
              <KeyValue label="Drug" value={entity_detected.drug} />
            </Section>
          )}

          {/* Pipeline stages */}
          {pipeline_stages?.length > 0 && (
            <Section title="Pipeline Stages">
              <p className="reasoning-stages">
                {pipeline_stages.map((s, i) => (
                  <span key={i} className="stage-chip">
                    {s}
                  </span>
                ))}
              </p>
            </Section>
          )}

          {/* Rationale plan */}
          {rationale_plan?.length > 0 && (
            <Section title="Rationale Plan">
              <ol className="reasoning-plan-list">
                {rationale_plan.map((step, i) => (
                  <li key={i}>
                    <strong>{step.step || `Step ${i + 1}`}:</strong>{' '}
                    {step.action || JSON.stringify(step)}
                  </li>
                ))}
              </ol>
            </Section>
          )}

          {/* Causal ranking */}
          {causal_ranking?.length > 0 && (
            <Section title="Top Causal Paths">
              <table className="reasoning-table">
                <thead>
                  <tr>
                    <th>Ingredient</th>
                    <th>Compound</th>
                    <th>Drug</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {causal_ranking.slice(0, 5).map((p, i) => (
                    <tr key={i}>
                      <td>{p.ingredient || '—'}</td>
                      <td>{p.compound || '—'}</td>
                      <td>{p.drug || '—'}</td>
                      <td>{typeof p.causal_score === 'number' ? p.causal_score.toFixed(3) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {causal_summary && (
                <p className="reasoning-note">
                  Avg causal score:{' '}
                  {typeof causal_summary.avg_causal_score === 'number'
                    ? causal_summary.avg_causal_score.toFixed(3)
                    : '—'}
                  {' · '}Top path:{' '}
                  {typeof causal_summary.top_causal_score === 'number'
                    ? causal_summary.top_causal_score.toFixed(3)
                    : '—'}
                </p>
              )}
            </Section>
          )}

          {/* Dosage validation */}
          {dosage_validation && (
            <Section title="Dosage Validation">
              <KeyValue
                label="Alignment"
                value={
                  typeof dosage_validation.overall_alignment_score === 'number'
                    ? `${(dosage_validation.overall_alignment_score * 100).toFixed(0)}%`
                    : null
                }
              />
              {dosage_validation.warnings?.length > 0 && (
                <ul className="reasoning-warnings">
                  {dosage_validation.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              )}
            </Section>
          )}

          {/* Faith alignment */}
          {faith_alignment_score != null && (
            <Section title="Faith-Science Alignment">
              <KeyValue
                label="Score"
                value={`${(faith_alignment_score * 100).toFixed(0)}%`}
              />
              {faith_alignment_notes && (
                <p className="reasoning-note">{faith_alignment_notes}</p>
              )}
            </Section>
          )}

          {/* Confidence breakdown */}
          {confidence_breakdown && (
            <Section title="Confidence Breakdown">
              <KeyValue
                label="Mapping Strength"
                value={confidence_breakdown.mapping_strength}
              />
              <KeyValue
                label="Path Coverage"
                value={confidence_breakdown.path_coverage}
              />
              <KeyValue
                label="Hadith Presence"
                value={confidence_breakdown.hadith_presence}
              />
              <KeyValue
                label="Faith Alignment"
                value={confidence_breakdown.faith_alignment}
              />
              <KeyValue
                label="Causal Avg"
                value={confidence_breakdown.causal_avg}
              />
              <KeyValue
                label="Dosage Alignment"
                value={confidence_breakdown.dosage_alignment}
              />
            </Section>
          )}

          {/* Multi-hop */}
          {multi_hop_activated && (
            <Section title="Multi-Hop Discovery">
              <p className="reasoning-note">
                Multi-hop exploratory search was activated for this query.
              </p>
            </Section>
          )}

          {/* Evaluation metrics */}
          {evaluation_metrics && (
            <Section title="Evaluation Metrics">
              <KeyValue label="Combined Score" value={evaluation_metrics.combined_score} />
              {evaluation_metrics.white_box && (
                <KeyValue
                  label="White-Box"
                  value={evaluation_metrics.white_box.composite_white_score}
                />
              )}
              {evaluation_metrics.black_box && (
                <KeyValue
                  label="Black-Box"
                  value={evaluation_metrics.black_box.composite_black_score}
                />
              )}
            </Section>
          )}
        </div>
      )}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="reasoning-section">
      <h4 className="reasoning-section-title">{title}</h4>
      {children}
    </div>
  );
}

function KeyValue({ label, value }) {
  if (value == null) return null;
  const display =
    typeof value === 'number'
      ? Number.isInteger(value)
        ? value
        : value.toFixed(3)
      : String(value);
  return (
    <p className="reasoning-kv">
      <span className="reasoning-kv-label">{label}:</span>{' '}
      <span className="reasoning-kv-value">{display}</span>
    </p>
  );
}

export default ReasoningPanel;
