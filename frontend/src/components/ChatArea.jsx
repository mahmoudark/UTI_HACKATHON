import React, { useState, useRef, useEffect } from 'react';
import { MenuIcon, SendIcon, ClinicalIcon } from './Icons.jsx';
import ChatMessage from './ChatMessage.jsx';

const SAMPLE_PROMPTS = [
  "What antibiotics are recommended for men aged 16 years and over?",
  "What is the recommended antibiotic treatment for pregnant women?",
  "What antibiotics are recommended for non-pregnant women aged 16 years and over?",
  "What self-care advice should be given for lower UTI?",
];

export default function ChatArea({
  chat,
  onSendMessage,
  onToggleSidebar,
  isLoading,
  healthStatus,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const messages = chat?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    onSendMessage(text);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-container">
      {/* Top Bar */}
      <header className="chat-header">
        <div className="header-left">
          <button
            type="button"
            className="menu-toggle-btn"
            onClick={onToggleSidebar}
            aria-label="Open menu"
          >
            <MenuIcon className="w-5 h-5" />
          </button>
          <div className="header-title-box">
            <span className="header-title">{chat?.title || 'UTI Clinical Decision Support'}</span>
            <span className="header-badge">NICE NG109</span>
          </div>
        </div>

        <div className="header-right">
          <span
            className={`status-pill ${
              healthStatus?.status === 'ok' ? 'status-online' : 'status-offline'
            }`}
          >
            <span className="status-dot" />
            {healthStatus?.status === 'ok' ? 'System Online' : 'Offline'}
          </span>
        </div>
      </header>

      {/* Messages Stream */}
      <div className="chat-scroll-area">
        <div className="chat-content-wrapper">
          {messages.length === 0 ? (
            /* Clear Empty State */
            <div className="empty-state">
              <div className="empty-icon-box">
                <ClinicalIcon className="w-5 h-5" />
              </div>
              <h2 className="empty-title">How can I help with UTI guidance today?</h2>
              <p className="empty-subtitle">
                Ask a clinical question to retrieve evidence-grounded recommendations from the official NICE NG109 guideline.
              </p>

              <div className="prompt-chips-grid">
                {SAMPLE_PROMPTS.map((promptText, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="prompt-chip-btn"
                    onClick={() => {
                      if (!isLoading) onSendMessage(promptText);
                    }}
                  >
                    <span>{promptText}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Message Bubbles */
            <div className="messages-list">
              {messages.map((msg, index) => (
                <ChatMessage
                  key={msg.id || index}
                  message={msg}
                  onRetry={
                    msg.error
                      ? () => {
                          const prev = messages[index - 1];
                          if (prev && prev.role === 'user') onSendMessage(prev.content);
                        }
                      : null
                  }
                />
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="message-row message-assistant">
                  <div className="assistant-avatar">
                    <ClinicalIcon size={14} />
                  </div>
                  <div className="assistant-card loading-card">
                    <div className="loading-dots">
                      <span className="dot" />
                      <span className="dot" />
                      <span className="dot" />
                    </div>
                    <span className="loading-text">Retrieving evidence from NICE NG109...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Bottom Large Chat Composer */}
      <div className="composer-container">
        <div className="composer-wrapper">
          <form onSubmit={handleSubmit} className="composer-form">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about UTI treatment..."
              rows={1}
              className="composer-input"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="composer-btn-send"
              aria-label="Send query"
            >
              <SendIcon className="w-4 h-4" />
            </button>
          </form>
          <div className="composer-footer">
            <span>Grounded in NICE NG109 guideline. For clinical decision support only.</span>
          </div>
        </div>
      </div>
    </div>
  );
}