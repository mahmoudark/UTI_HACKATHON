import React, { useState, useEffect, useCallback } from 'react';
import { getHealth, postAnswer } from './api.js';
import Sidebar from './components/Sidebar.jsx';
import ChatArea from './components/ChatArea.jsx';
import SourcesModal from './components/SourcesModal.jsx';
import SettingsModal from './components/SettingsModal.jsx';

const STORAGE_KEY_CHATS = 'uti_cds_chats_stitch_v3';
const STORAGE_KEY_ACTIVE_ID = 'uti_cds_active_chat_stitch_v3';
const STORAGE_KEY_SETTINGS = 'uti_cds_settings_stitch_v3';

function createNewChat() {
  return {
    id: `chat_${Date.now()}`,
    title: 'New Consultation',
    createdAt: Date.now(),
    messages: [],
  };
}

export default function App() {
  const [chats, setChats] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_CHATS);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (_) {}
    return [createNewChat()];
  });

  const [activeChatId, setActiveChatId] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_ACTIVE_ID);
      if (saved) return saved;
    } catch (_) {}
    return chats[0]?.id || null;
  });

  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_SETTINGS);
      if (saved) return JSON.parse(saved);
    } catch (_) {}
    return { theme: 'light' };
  });

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);

  // Sync activeChatId if invalid
  useEffect(() => {
    if (!chats.find((c) => c.id === activeChatId)) {
      if (chats.length > 0) {
        setActiveChatId(chats[0].id);
      } else {
        const fresh = createNewChat();
        setChats([fresh]);
        setActiveChatId(fresh.id);
      }
    }
  }, [chats, activeChatId]);

  // Persist chats
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_CHATS, JSON.stringify(chats));
    } catch (_) {}
  }, [chats]);

  // Persist active ID
  useEffect(() => {
    if (activeChatId) {
      try {
        localStorage.setItem(STORAGE_KEY_ACTIVE_ID, activeChatId);
      } catch (_) {}
    }
  }, [activeChatId]);

  // Persist & apply settings / theme
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(settings));
    } catch (_) {}

    const isDark =
      settings.theme === 'dark' ||
      (settings.theme === 'system' &&
        window.matchMedia('(prefers-color-scheme: dark)').matches);

    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [settings]);

  // Check health periodically
  useEffect(() => {
    const check = () => {
      getHealth()
        .then(setHealthStatus)
        .catch(() => setHealthStatus({ status: 'offline' }));
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  const activeChat = chats.find((c) => c.id === activeChatId) || chats[0];

  const handleNewChat = useCallback(() => {
    const fresh = createNewChat();
    setChats((prev) => [fresh, ...prev]);
    setActiveChatId(fresh.id);
    setSidebarOpen(false);
  }, []);

  const handleDeleteChat = useCallback((chatId) => {
    setChats((prev) => {
      const filtered = prev.filter((c) => c.id !== chatId);
      return filtered.length === 0 ? [createNewChat()] : filtered;
    });
  }, []);

  const handleClearAllChats = useCallback(() => {
    const fresh = createNewChat();
    setChats([fresh]);
    setActiveChatId(fresh.id);
  }, []);

  const handleSendMessage = async (queryText) => {
    if (!queryText.trim() || isLoading || !activeChatId) return;

    const userMessage = {
      id: `u_${Date.now()}`,
      role: 'user',
      content: queryText,
    };

    setChats((prev) =>
      prev.map((c) => {
        if (c.id === activeChatId) {
          const isFirst = c.messages.length === 0;
          const newTitle = isFirst
            ? queryText.length > 32
              ? `${queryText.substring(0, 32)}...`
              : queryText
            : c.title;
          return {
            ...c,
            title: newTitle,
            messages: [...c.messages, userMessage],
          };
        }
        return c;
      })
    );

    setIsLoading(true);

    try {
      const res = await postAnswer(queryText);
      const assistantMessage = {
        id: `a_${Date.now()}`,
        role: 'assistant',
        status: res.status,
        data: res,
      };

      setChats((prev) =>
        prev.map((c) =>
          c.id === activeChatId
            ? { ...c, messages: [...c.messages, assistantMessage] }
            : c
        )
      );
    } catch (err) {
      const errMessage = {
        id: `err_${Date.now()}`,
        role: 'assistant',
        status: 'ERROR',
        error: err.message || 'Could not connect to clinical backend.',
      };

      setChats((prev) =>
        prev.map((c) =>
          c.id === activeChatId
            ? { ...c, messages: [...c.messages, errMessage] }
            : c
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="stitch-app">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={(id) => setActiveChatId(id)}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onOpenSources={() => setSourcesOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <ChatArea
        chat={activeChat}
        onSendMessage={handleSendMessage}
        onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
        isLoading={isLoading}
        healthStatus={healthStatus}
      />

      <SourcesModal isOpen={sourcesOpen} onClose={() => setSourcesOpen(false)} />
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={(newVals) => setSettings((prev) => ({ ...prev, ...newVals }))}
        onClearAllChats={handleClearAllChats}
      />
    </div>
  );
}