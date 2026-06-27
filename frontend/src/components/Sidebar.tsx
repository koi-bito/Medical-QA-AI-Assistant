"use client";

import React from "react";
import { useAuth } from "@/context/AuthContext";
import { useChat } from "@/context/ChatContext";
import { PlusCircle, LogOut, MessageSquare, Trash2 } from "lucide-react";

export default function Sidebar({ onClose }: { onClose?: () => void }) {
  const { user, logout } = useAuth();
  const { conversations, activeConversationId, setActiveConversationId, deleteConversation } = useChat();

  return (
    <aside className="w-72 border-r border-white/30 bg-white/60 backdrop-blur-xl flex flex-col h-full shadow-[4px_0_24px_rgba(0,0,0,0.02)]">
      <div className="p-4">
        <button 
          onClick={() => {
            setActiveConversationId(null);
            if (onClose) onClose();
          }}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          <PlusCircle className="h-4 w-4" />
          New Chat
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {conversations.length === 0 ? (
          <p className="text-sm text-gray-500 text-center mt-4">No conversations yet</p>
        ) : (
          conversations.map((conv) => (
            <div 
              key={conv.id}
              onClick={() => {
                setActiveConversationId(conv.id);
                if (onClose) onClose();
              }}
              className={`group flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all duration-200 ${
                activeConversationId === conv.id 
                  ? "bg-white/90 text-blue-900 shadow-sm border border-white/50 font-medium" 
                  : "text-gray-700 hover:bg-white/60 border border-transparent"
              }`}
            >
              <div className="flex items-center gap-2 overflow-hidden">
                <MessageSquare className="h-4 w-4 flex-shrink-0" />
                <span className="text-sm truncate select-none" title={conv.title}>
                  {conv.title}
                </span>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(conv.id);
                }}
                className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity focus:outline-none p-1"
                title="Delete conversation"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))
        )}
      </div>
      
      <div className="p-4 border-t border-white/30 bg-white/40 flex items-center justify-between">
        <div className="flex flex-col overflow-hidden">
          <span className="text-sm font-medium text-gray-900 truncate">{user?.username}</span>
          <span className="text-xs text-gray-500 truncate">{user?.email}</span>
        </div>
        <button
          onClick={logout}
          className="p-2 text-gray-500 hover:text-red-600 rounded-lg hover:bg-white/80 transition-all duration-200"
          title="Logout"
        >
          <LogOut className="h-5 w-5" />
        </button>
      </div>
    </aside>
  );
}
