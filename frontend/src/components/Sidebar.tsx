"use client";

import React from "react";
import { useAuth } from "@/context/AuthContext";
import { useChat } from "@/context/ChatContext";
import { PlusCircle, LogOut, MessageSquare, Trash2 } from "lucide-react";

export default function Sidebar() {
  const { user, logout } = useAuth();
  const { conversations, activeConversationId, setActiveConversationId, deleteConversation } = useChat();

  return (
    <aside className="w-72 border-r border-gray-200 bg-gray-50 flex flex-col h-full">
      <div className="p-4">
        <button 
          onClick={() => setActiveConversationId(null)}
          className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
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
              onClick={() => setActiveConversationId(conv.id)}
              className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors ${
                activeConversationId === conv.id 
                  ? "bg-blue-100 text-blue-900" 
                  : "text-gray-700 hover:bg-gray-200"
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
      
      <div className="p-4 border-t border-gray-200 bg-gray-100 flex items-center justify-between">
        <div className="flex flex-col overflow-hidden">
          <span className="text-sm font-medium text-gray-900 truncate">{user?.username}</span>
          <span className="text-xs text-gray-500 truncate">{user?.email}</span>
        </div>
        <button
          onClick={logout}
          className="p-2 text-gray-500 hover:text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
          title="Logout"
        >
          <LogOut className="h-5 w-5" />
        </button>
      </div>
    </aside>
  );
}
