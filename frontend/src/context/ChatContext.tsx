"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import api from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

export interface Conversation {
  id: number;
  title: string;
  created_at: string;
}

interface ChatContextType {
  conversations: Conversation[];
  activeConversationId: number | null;
  setActiveConversationId: (id: number | null) => void;
  fetchConversations: () => Promise<void>;
  deleteConversation: (id: number) => Promise<void>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const { user } = useAuth(); // To fetch when user logs in

  const fetchConversations = async () => {
    try {
      const response = await api.get("/conversations/");
      setConversations(response.data);
    } catch (error) {
      console.error("Failed to fetch conversations:", error);
    }
  };

  const deleteConversation = async (id: number) => {
    try {
      await api.delete(`/conversations/${id}`);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        setActiveConversationId(null);
      }
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  };

  useEffect(() => {
    if (user) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      fetchConversations();
    } else {
      setConversations([]);
      setActiveConversationId(null);
    }
  }, [user]);

  return (
    <ChatContext.Provider
      value={{
        conversations,
        activeConversationId,
        setActiveConversationId,
        fetchConversations,
        deleteConversation,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
}
