"use client";

import React from "react";
import { useAuth } from "@/context/AuthContext";
import { useChat } from "@/context/ChatContext";
import { PlusCircle, LogOut, MessageSquare, Trash2, Heart } from "lucide-react";

export default function Sidebar({ onClose }: { onClose?: () => void }) {
  const { user, logout } = useAuth();
  const { conversations, activeConversationId, setActiveConversationId, deleteConversation } = useChat();

  return (
    <aside
      style={{ backgroundColor: "#faf7f2", borderRight: "1px solid #ede8df" }}
      className="w-72 flex flex-col h-full"
    >
      {/* ── Brand Header ── */}
      <div
        style={{ borderBottom: "1px solid #ede8df", background: "linear-gradient(135deg, #fffdf9 0%, #faf0ef 100%)" }}
        className="p-5 flex items-center gap-3"
      >
        <div
          style={{ background: "linear-gradient(135deg, #c8102e 0%, #8b0a1e 100%)" }}
          className="w-9 h-9 rounded-xl flex items-center justify-center shadow-md flex-shrink-0"
        >
          <Heart className="h-5 w-5 text-white fill-white" />
        </div>
        <div>
          <p style={{ color: "#2e261d" }} className="text-sm font-bold tracking-tight leading-none">
            MedQA Assistant
          </p>
          <p style={{ color: "#9b8f85" }} className="text-[11px] mt-0.5">
            AI-Powered Medical QA
          </p>
        </div>
      </div>

      {/* ── New Chat Button ── */}
      <div className="px-4 pt-4 pb-2">
        <button
          onClick={() => {
            setActiveConversationId(null);
            if (onClose) onClose();
          }}
          style={{
            background: "linear-gradient(135deg, #c8102e 0%, #a80d25 100%)",
            boxShadow: "0 4px 12px rgba(200,16,46,0.25)",
          }}
          className="flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold text-white transition-all duration-200 hover:scale-[1.02] hover:shadow-lg active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-2"
        >
          <PlusCircle className="h-4 w-4" />
          New Chat
        </button>
      </div>

      {/* ── Conversations ── */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
        {conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center mt-12 px-4 gap-3">
            <div
              style={{ background: "#f0f7ee", border: "1px solid #d6edcf" }}
              className="w-12 h-12 rounded-2xl flex items-center justify-center"
            >
              <MessageSquare style={{ color: "#4a7c59" }} className="h-6 w-6" />
            </div>
            <p style={{ color: "#9b8f85" }} className="text-sm text-center leading-relaxed">
              No conversations yet.
              <br />Start a new chat above!
            </p>
          </div>
        ) : (
          conversations.map((conv) => {
            const isActive = activeConversationId === conv.id;
            return (
              <div
                key={conv.id}
                onClick={() => {
                  setActiveConversationId(conv.id);
                  if (onClose) onClose();
                }}
                style={
                  isActive
                    ? { background: "#fff1f1", border: "1px solid #ffdede", color: "#8b0a1e" }
                    : { background: "transparent", border: "1px solid transparent", color: "#5c5045" }
                }
                className="group flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-150 hover:bg-[#fffdf9] hover:border-[#ede8df]"
              >
                <div className="flex items-center gap-2.5 overflow-hidden min-w-0">
                  <MessageSquare
                    style={{ color: isActive ? "#c8102e" : "#9b8f85" }}
                    className="h-3.5 w-3.5 flex-shrink-0 transition-colors"
                  />
                  <span className="text-sm truncate select-none font-medium" title={conv.title}>
                    {conv.title}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conv.id);
                  }}
                  style={{ color: "#c8a98a" }}
                  className="opacity-0 group-hover:opacity-100 transition-all duration-150 hover:!text-red-600 p-1 rounded-lg hover:bg-red-50 flex-shrink-0 ml-1"
                  title="Delete conversation"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* ── User Footer ── */}
      <div
        style={{ borderTop: "1px solid #ede8df", background: "#f5f0e8" }}
        className="p-4 flex items-center gap-3"
      >
        <div
          style={{ background: "linear-gradient(135deg, #4a7c59 0%, #2d4e38 100%)" }}
          className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm"
        >
          <span className="text-white text-xs font-bold uppercase">
            {user?.username?.[0] ?? "U"}
          </span>
        </div>
        <div className="flex-1 overflow-hidden min-w-0">
          <p style={{ color: "#2e261d" }} className="text-sm font-semibold truncate leading-tight">
            {user?.username}
          </p>
          <p style={{ color: "#9b8f85" }} className="text-[11px] truncate leading-tight mt-0.5">
            {user?.email}
          </p>
        </div>
        <button
          onClick={logout}
          style={{ color: "#9b8f85" }}
          className="p-2 rounded-lg hover:bg-red-50 hover:!text-red-600 transition-all duration-150 flex-shrink-0"
          title="Logout"
        >
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </aside>
  );
}
