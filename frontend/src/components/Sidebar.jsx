import React from 'react';
import {
  ClinicalIcon,
  PlusIcon,
  ChatIcon,
  BookIcon,
  SettingsIcon,
  TrashIcon,
  UserIcon,
  XIcon,
} from './Icons.jsx';

export default function Sidebar({
  isOpen,
  onClose,
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onOpenSources,
  onOpenSettings,
}) {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && <div className="sidebar-backdrop" onClick={onClose} />}

      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        {/* Header */}
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-icon">
              <ClinicalIcon className="w-4 h-4" />
            </div>
            <div>
              <h1 className="brand-title">UTI Clinical Decision Support</h1>
              <p className="brand-subtitle">Clinical AI Workspace</p>
            </div>
          </div>
          <button
            type="button"
            className="mobile-close-btn"
            onClick={onClose}
            aria-label="Close menu"
          >
            <XIcon className="w-4 h-4" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="sidebar-new-chat">
          <button type="button" className="btn-new-chat" onClick={onNewChat}>
            <PlusIcon className="w-4 h-4" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Recent Chats Section */}
        <div className="sidebar-history">
          <div className="history-label">Recent Chats</div>
          <div className="history-list">
            {chats.length === 0 ? (
              <div className="history-empty">No previous chats</div>
            ) : (
              chats.map((chat) => {
                const isActive = chat.id === activeChatId;
                return (
                  <div
                    key={chat.id}
                    className={`history-item ${isActive ? 'active' : ''}`}
                    onClick={() => {
                      onSelectChat(chat.id);
                      if (window.innerWidth < 768) onClose();
                    }}
                  >
                    <ChatIcon className="w-3.5 h-3.5 item-icon" />
                    <span className="item-title" title={chat.title || 'New Consultation'}>
                      {chat.title || 'New Consultation'}
                    </span>
                    <button
                      type="button"
                      className="item-delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChat(chat.id);
                      }}
                      title="Delete chat"
                    >
                      <TrashIcon className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Bottom Navigation */}
        <div className="sidebar-nav">
          <button type="button" className="nav-item" onClick={onOpenSources}>
            <BookIcon className="w-4 h-4" />
            <span>Sources</span>
          </button>
          <button type="button" className="nav-item" onClick={onOpenSettings}>
            <SettingsIcon className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>

        {/* User Profile Area */}
        <div className="sidebar-profile">
          <div className="profile-avatar">
            <UserIcon className="w-3.5 h-3.5" />
          </div>
          <div className="profile-info">
            <span className="profile-name">Dr. Clinical User</span>
            <span className="profile-role">NICE NG109 Decision Support</span>
          </div>
        </div>
      </aside>
    </>
  );
}