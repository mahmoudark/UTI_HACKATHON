import React, { useState } from 'react';
import {
  ClinicalIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CopyIcon,
  CheckIcon,
} from './Icons.jsx';

export default function ChatMessage({ message, onRetry }) {
  const isUser = message.role === 'user';
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const text = message.data?.answer || message.content || '';
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="message-row message-user">
        <div className="user-bubble">
          <p>{message.content}</p>
        </div>
      </div>
    );
  }

  const { status, data, error } = message;
  const isAnswered = status === 'ANSWERED' && data?.answer;
  const isRefused = status === 'REFUSED';
  const source = data?.source;
  const results = data?.results || [];

  const formatAnswerContent = (answer) => {
    if (!answer) return null;
    const lines = answer.split('\n');

    return lines.map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) return <div key={idx} className="line-break" />;

      // Preamble line
      if (
        trimmed.startsWith('The guideline recommends:') ||
        trimmed.startsWith('The guideline suggests:') ||
        trimmed.startsWith('Limited evidence found')
      ) {
        return (
          <p key={idx} className="answer-lead-text">
            {trimmed}
          </p>
        );
      }

      // According to source line
      if (trimmed.startsWith('According to ')) {
        return (
          <p key={idx} className="answer-according-text">
            {trimmed}
          </p>
        );
      }

      // First choices header - visually prominent
      if (
        trimmed.toLowerCase().startsWith('first choice') ||
        trimmed.toLowerCase().startsWith('first choices')
      ) {
        return (
          <h4 key={idx} className="answer-section-first-choice">
            {trimmed}
          </h4>
        );
      }

      // Second choices header
      if (
        trimmed.toLowerCase().startsWith('second choice') ||
        trimmed.toLowerCase().startsWith('second choices')
      ) {
        return (
          <h4 key={idx} className="answer-section-second-choice">
            {trimmed}
          </h4>
        );
      }

      // Other section title ending with colon
      if (trimmed.endsWith(':')) {
        return (
          <h4 key={idx} className="answer-section-title">
            {trimmed}
          </h4>
        );
      }

      // Bullet points - scannable drug names & dosages
      if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
        const rawItem = trimmed.substring(2);
        const colonIdx = rawItem.indexOf(':');

        if (colonIdx > 0 && colonIdx < 40) {
          const drugPart = rawItem.substring(0, colonIdx);
          const dosagePart = rawItem.substring(colonIdx + 1);
          return (
            <li key={idx} className="answer-bullet-item">
              <strong className="drug-name">{drugPart}:</strong>
              <span className="dosage-text">{dosagePart}</span>
            </li>
          );
        }

        return (
          <li key={idx} className="answer-bullet-item">
            <span className="bullet-text">{rawItem}</span>
          </li>
        );
      }

      // Notes / precautions
      if (
        trimmed.startsWith('Note:') ||
        trimmed.startsWith('See the BNF') ||
        trimmed.startsWith('Avoid at term')
      ) {
        return (
          <p key={idx} className="answer-note-text">
            {trimmed}
          </p>
        );
      }

      return (
        <p key={idx} className="answer-p">
          {trimmed}
        </p>
      );
    });
  };

  const formatSourceCitation = (src) => {
    if (!src) return null;
    const title = src.title || src.source_id || '';
    const pages = src.pages ? (Array.isArray(src.pages) ? src.pages.join(', ') : src.pages) : null;

    const tableMatch = title.match(/Table\s+(\d+)/i);
    const tablePart = tableMatch ? `Table ${tableMatch[1]}` : (title || 'Guideline Reference');

    const items = ['NICE NG109'];
    if (tablePart && tablePart !== 'NICE NG109') items.push(tablePart);
    if (pages) items.push(`Page ${pages}`);

    return items.join(' • ');
  };

  return (
    <div className="message-row message-assistant">
      <div className="assistant-avatar">
        <ClinicalIcon size={14} />
      </div>

      <div className="assistant-card">
        {/* Answered State */}
        {isAnswered && (
          <>
            {/* Header Badge */}
            <div className="card-header">
              <div className="badge-group">
                <span className="badge-grounded">NICE NG109 Grounded</span>
                {data.confidence && (
                  <span className={`badge-confidence badge-${data.confidence.toLowerCase()}`}>
                    {data.confidence} Confidence
                  </span>
                )}
              </div>
              <button
                type="button"
                className="btn-copy"
                onClick={handleCopy}
                title="Copy answer"
              >
                {copied ? (
                  <>
                    <CheckIcon size={12} className="text-green" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <CopyIcon size={12} />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>

            {/* Main Answer - The Primary Focus with clear hierarchy */}
            <div className="answer-text">{formatAnswerContent(data.answer)}</div>

            {/* Source Citation */}
            {source && (
              <div className="citation-row">
                <span className="citation-label">Source:</span>
                <span className="citation-value">{formatSourceCitation(source)}</span>
              </div>
            )}

            {/* Evidence Used - Improved 'View supporting evidence' Collapsible Section */}
            {results && results.length > 0 && (
              <div className="evidence-section">
                <button
                  type="button"
                  className="evidence-toggle-btn"
                  onClick={() => setEvidenceOpen(!evidenceOpen)}
                  aria-expanded={evidenceOpen}
                >
                  {evidenceOpen ? (
                    <ChevronDownIcon size={12} />
                  ) : (
                    <ChevronRightIcon size={12} />
                  )}
                  <span>
                    {evidenceOpen ? 'Hide supporting evidence' : 'View supporting evidence'}
                  </span>
                </button>

                {evidenceOpen && (
                  <div className="evidence-dropdown">
                    {results.map((item, idx) => {
                      const meta = item.metadata || {};
                      return (
                        <div key={idx} className="evidence-item">
                          <div className="evidence-item-header">
                            <div className="evidence-header-left">
                              <span className="evidence-doc-badge">NICE NG109</span>
                              <span className="evidence-section-name">
                                {meta.title || meta.source_id || `Section ${idx + 1}`}
                              </span>
                            </div>
                            {meta.pages && (
                              <span className="evidence-page-tag">Page {meta.pages}</span>
                            )}
                          </div>
                          <p className="evidence-excerpt">{item.document || item.text}</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* Refused State */}
        {isRefused && (
          <div className="refusal-box">
            <div className="refusal-header">
              <span className="badge-warning">Guideline Scope Notice</span>
            </div>
            <p className="refusal-message">
              {data?.reason || 'No clinical recommendation could be generated from available evidence.'}
            </p>
            <p className="refusal-subtext">
              Answers are grounded strictly in NICE NG109 (Lower UTI) guideline recommendations for non-pregnant women, pregnant women, men, and children.
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="error-box">
            <span className="badge-error">Backend Connection Error</span>
            <p className="error-message">{error}</p>
            {onRetry && (
              <button type="button" className="btn-retry" onClick={onRetry}>
                Retry Query
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}