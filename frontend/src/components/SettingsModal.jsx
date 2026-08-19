import React from 'react';
import { XIcon, SettingsIcon, TrashIcon } from './Icons.jsx';

export default function SettingsModal({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
  onClearAllChats,
}) {
  if (!isOpen) return null;

  const { theme = 'system' } = settings;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box modal-sm" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-title">
            <SettingsIcon className="w-4 h-4" />
            <span>Settings</span>
          </div>
          <button type="button" className="btn-close" onClick={onClose}>
            <XIcon className="w-4 h-4" />
          </button>
        </div>

        <div className="modal-body">
          {/* Theme */}
          <div className="setting-group">
            <label className="setting-label">Appearance</label>
            <div className="theme-toggle-row">
              <button
                type="button"
                className={`theme-btn ${theme === 'light' ? 'active' : ''}`}
                onClick={() => onUpdateSettings({ theme: 'light' })}
              >
                Light
              </button>
              <button
                type="button"
                className={`theme-btn ${theme === 'dark' ? 'active' : ''}`}
                onClick={() => onUpdateSettings({ theme: 'dark' })}
              >
                Dark
              </button>
              <button
                type="button"
                className={`theme-btn ${theme === 'system' ? 'active' : ''}`}
                onClick={() => onUpdateSettings({ theme: 'system' })}
              >
                System
              </button>
            </div>
          </div>

          {/* History */}
          <div className="setting-group">
            <label className="setting-label">Local Storage</label>
            <button
              type="button"
              className="btn-clear-history"
              onClick={() => {
                if (window.confirm('Clear all conversation history?')) {
                  onClearAllChats();
                  onClose();
                }
              }}
            >
              <TrashIcon className="w-3.5 h-3.5" />
              <span>Clear All Chat History</span>
            </button>
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn-done" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}