"use client";

import React, { useState, useRef, useEffect } from "react";
import api from "@/lib/api";
import ReactMarkdown from "react-markdown";
import {
  Send,
  User,
  Bot,
  AlertTriangle,
  Info,
  ChevronDown,
  ThumbsUp,
  ThumbsDown,
  Stethoscope,
} from "lucide-react";
import toast from "react-hot-toast";
import { useChat } from "@/context/ChatContext";

interface Source {
  chunk_id: string;
  topic: string;
  text: string;
  distance: number;
}

interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  confidence?: string;
  feedback?: "up" | "down";
}

const confidenceStyle = (level?: string) => {
  const l = level?.toLowerCase();
  if (l === "high")   return { bg: "#f0f7ee", color: "#3a6347", border: "#d6edcf" };
  if (l === "medium") return { bg: "#fefce8", color: "#854d0e", border: "#fef08a" };
  return                     { bg: "#fff1f1", color: "#9b1c1c", border: "#ffdede" };
};

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { activeConversationId, setActiveConversationId, fetchConversations } = useChat();
  const justCreatedRef = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => { scrollToBottom(); }, [messages, loading]);

  // Load messages when active conversation changes
  useEffect(() => {
    if (activeConversationId) {
      if (justCreatedRef.current) {
        justCreatedRef.current = false;
        return;
      }
      const fetchMessages = async () => {
        setLoading(true);
        try {
          const response = await api.get(`/conversations/${activeConversationId}`);
          const formattedMessages = response.data.messages.map(
            (m: { id: number; role: "user" | "assistant"; content: string; sources: string | null; confidence: string | null }) => ({
              id: m.id,
              role: m.role,
              content: m.content,
              sources: m.sources ? JSON.parse(m.sources) : undefined,
              confidence: m.confidence || undefined,
            }),
          );
          setMessages(formattedMessages);
        } catch {
          toast.error("Failed to load conversation.");
        } finally {
          setLoading(false);
        }
      };
      fetchMessages();
    } else {
      setMessages([]);
    }
  }, [activeConversationId]);

  const handleFeedback = async (messageId: number, rating: "up" | "down", index: number) => {
    try {
      await api.post("/feedback", { message_id: messageId, rating });
      setMessages((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], feedback: rating };
        return next;
      });
      toast.success("Feedback submitted!");
    } catch {
      toast.error("Failed to submit feedback.");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const response = await api.post("/ask", {
        question: userMessage,
        conversation_id: activeConversationId || undefined,
      });

      setMessages((prev) => [
        ...prev,
        {
          id: response.data.message_id,
          role: "assistant",
          content: response.data.answer,
          sources: response.data.sources,
          confidence: response.data.confidence,
        },
      ]);

      if (!activeConversationId && response.data.conversation_id) {
        justCreatedRef.current = true;
        setActiveConversationId(response.data.conversation_id);
        fetchConversations();
      }
    } catch {
      toast.error("Failed to get an answer. Please try again.");
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative" style={{ backgroundColor: "#faf7f2" }}>

      {/* ── Disclaimer Banner ── */}
      <div
        style={{ backgroundColor: "#fff8f0", borderBottom: "1px solid #ede8df" }}
        className="flex items-center justify-center gap-2 px-4 py-2 flex-shrink-0 z-10"
      >
        <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" style={{ color: "#c8102e" }} />
        <p className="text-xs font-medium" style={{ color: "#5c5045" }}>
          <span className="font-semibold" style={{ color: "#a80d25" }}>Medical Disclaimer: </span>
          AI can make mistakes. Always consult a qualified healthcare provider for medical advice.
        </p>
      </div>

      {/* ── Messages Area ── */}
      {/* Solid background — the core fix for the "hole" problem */}
      <div
        className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 pb-36"
        style={{ backgroundColor: "#faf7f2" }}
      >
        <div className="max-w-3xl mx-auto space-y-6">

          {/* Empty state */}
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center min-h-[420px] gap-5">
              <div
                style={{ background: "linear-gradient(135deg, #fff1f1 0%, #faf0ef 100%)", border: "2px solid #ffdede" }}
                className="w-24 h-24 rounded-3xl flex items-center justify-center shadow-sm"
              >
                <Stethoscope style={{ color: "#c8102e" }} className="h-12 w-12" />
              </div>
              <div className="text-center">
                <h2 className="text-2xl font-bold tracking-tight" style={{ color: "#2e261d" }}>
                  Medical QA Assistant
                </h2>
                <p className="text-sm mt-2 max-w-sm leading-relaxed" style={{ color: "#9b8f85" }}>
                  Ask any medical question. I'll search trusted clinical literature and provide a sourced, evidence-based answer.
                </p>
              </div>
              {/* Suggestion chips */}
              <div className="flex flex-wrap justify-center gap-2 mt-2">
                {[
                  "What are symptoms of diabetes?",
                  "Is ibuprofen safe daily?",
                  "Difference between cold and flu?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => setInput(q)}
                    style={{ background: "#fffdf9", border: "1px solid #ede8df", color: "#5c5045" }}
                    className="text-xs px-3 py-1.5 rounded-full hover:border-red-300 hover:text-[#c8102e] transition-all duration-150"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message bubbles */}
          {messages.map((msg, index) => (
            <div
              key={index}
              className="flex gap-3 animate-in fade-in slide-in-from-bottom-3 duration-300"
            >
              {/* Avatar */}
              <div className="flex-shrink-0 mt-0.5">
                {msg.role === "user" ? (
                  <div
                    style={{ background: "linear-gradient(135deg, #c8102e 0%, #8b0a1e 100%)", boxShadow: "0 2px 8px rgba(200,16,46,0.3)" }}
                    className="w-8 h-8 rounded-full flex items-center justify-center"
                  >
                    <User className="h-4 w-4 text-white" />
                  </div>
                ) : (
                  <div
                    style={{ background: "linear-gradient(135deg, #4a7c59 0%, #2d4e38 100%)", boxShadow: "0 2px 8px rgba(74,124,89,0.3)" }}
                    className="w-8 h-8 rounded-full flex items-center justify-center"
                  >
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                )}
              </div>

              {/* Bubble */}
              <div
                className="flex-1 min-w-0 rounded-2xl px-5 py-4 shadow-sm"
                style={
                  msg.role === "user"
                    ? {
                        background: "linear-gradient(135deg, #c8102e 0%, #a80d25 100%)",
                        color: "#fff",
                        borderTopLeftRadius: "4px",
                        boxShadow: "0 4px 16px rgba(200,16,46,0.15)",
                      }
                    : {
                        backgroundColor: "#fffdf9",
                        border: "1px solid #ede8df",
                        color: "#2e261d",
                        borderTopLeftRadius: "4px",
                        boxShadow: "0 2px 12px rgba(46,38,29,0.05)",
                      }
                }
              >
                {/* Content */}
                <div
                  className={`prose prose-sm max-w-none break-words ${
                    msg.role === "user"
                      ? "prose-p:text-white prose-p:m-0 text-white"
                      : "prose-headings:text-[#2e261d] prose-p:text-[#3d3129]"
                  }`}
                >
                  {msg.role === "user" ? (
                    <p className="whitespace-pre-wrap m-0 font-medium">{msg.content}</p>
                  ) : (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  )}
                </div>

                {/* Sources & Confidence (assistant only) */}
                {msg.role === "assistant" && msg.sources !== undefined && (
                  <div
                    style={{ borderTop: "1px solid #ede8df" }}
                    className="mt-4 pt-4 space-y-3"
                  >
                    {/* Confidence + Feedback row */}
                    <div className="flex items-center justify-between">
                      {/* Confidence badge */}
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "#9b8f85" }}>
                          Confidence
                        </span>
                        {msg.confidence && (() => {
                          const s = confidenceStyle(msg.confidence);
                          return (
                            <span
                              className="text-xs px-2.5 py-0.5 rounded-full font-semibold border"
                              style={{ background: s.bg, color: s.color, borderColor: s.border }}
                            >
                              {msg.confidence}
                            </span>
                          );
                        })()}
                      </div>

                      {/* Thumbs feedback */}
                      {msg.id && (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => handleFeedback(msg.id!, "up", index)}
                            title="Helpful"
                            className="p-1.5 rounded-lg transition-all duration-150"
                            style={
                              msg.feedback === "up"
                                ? { background: "#f0f7ee", color: "#4a7c59" }
                                : { color: "#c8a98a" }
                            }
                          >
                            <ThumbsUp className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={() => handleFeedback(msg.id!, "down", index)}
                            title="Not Helpful"
                            className="p-1.5 rounded-lg transition-all duration-150"
                            style={
                              msg.feedback === "down"
                                ? { background: "#fff1f1", color: "#c8102e" }
                                : { color: "#c8a98a" }
                            }
                          >
                            <ThumbsDown className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Sources collapsible */}
                    {msg.sources.length > 0 && (
                      <details className="group">
                        <summary
                          style={{ color: "#5c5045" }}
                          className="text-xs font-semibold cursor-pointer hover:text-[#c8102e] flex items-center gap-1.5 list-none select-none transition-colors"
                        >
                          <Info className="h-3.5 w-3.5 flex-shrink-0" />
                          <span className="flex-1">View {msg.sources.length} source{msg.sources.length !== 1 ? "s" : ""}</span>
                          <ChevronDown className="h-3.5 w-3.5 transition-transform duration-200 group-open:rotate-180" />
                        </summary>
                        <div className="mt-3 space-y-3 pl-4" style={{ borderLeft: "2px solid #ede8df" }}>
                          {msg.sources.map((source, idx) => (
                            <div key={idx}>
                              <p className="text-xs font-semibold" style={{ color: "#3d3129" }}>{source.topic}</p>
                              <p className="text-xs mt-0.5 leading-relaxed line-clamp-3" style={{ color: "#9b8f85" }}>
                                {source.text}
                              </p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="flex gap-3">
              <div
                style={{ background: "linear-gradient(135deg, #4a7c59 0%, #2d4e38 100%)" }}
                className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-md mt-0.5"
              >
                <Bot className="h-4 w-4 text-white" />
              </div>
              <div
                style={{ backgroundColor: "#fffdf9", border: "1px solid #ede8df", borderTopLeftRadius: "4px" }}
                className="px-5 py-4 rounded-2xl flex items-center shadow-sm"
              >
                <div className="flex space-x-1.5">
                  <div className="w-2 h-2 rounded-full animate-bounce [animation-delay:-0.3s]" style={{ backgroundColor: "#4a7c59" }} />
                  <div className="w-2 h-2 rounded-full animate-bounce [animation-delay:-0.15s]" style={{ backgroundColor: "#4a7c59", opacity: 0.7 }} />
                  <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: "#4a7c59", opacity: 0.4 }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ── Input Bar ── */}
      {/* Solid gradient fade — no transparent hole at bottom */}
      <div
        className="absolute bottom-0 left-0 right-0 px-4 sm:px-8 pb-5 pt-10"
        style={{
          background: "linear-gradient(to top, #faf7f2 60%, transparent)",
        }}
      >
        <div className="max-w-3xl mx-auto">
          <form
            onSubmit={handleSubmit}
            className="relative flex items-end rounded-2xl overflow-hidden transition-all duration-200"
            style={{
              backgroundColor: "#fffdf9",
              border: "1.5px solid #ddd6c8",
              boxShadow: "0 4px 20px rgba(46,38,29,0.08)",
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask a medical question… (Shift+Enter for new line)"
              className="w-full resize-none border-0 outline-none py-4 pl-5 pr-14 text-sm font-medium bg-transparent placeholder:text-[#c8a98a]"
              style={{ color: "#2e261d", minHeight: "56px", maxHeight: "192px" }}
              rows={1}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="absolute right-2.5 bottom-2.5 p-2.5 rounded-xl text-white transition-all duration-200 hover:scale-105 active:scale-95 disabled:opacity-40 disabled:hover:scale-100"
              style={{
                background: "linear-gradient(135deg, #c8102e 0%, #a80d25 100%)",
                boxShadow: input.trim() ? "0 4px 12px rgba(200,16,46,0.35)" : "none",
              }}
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
          <p className="text-center text-[11px] mt-2" style={{ color: "#c8a98a" }}>
            AI-generated content. Always verify medical information with a healthcare professional.
          </p>
        </div>
      </div>
    </div>
  );
}
