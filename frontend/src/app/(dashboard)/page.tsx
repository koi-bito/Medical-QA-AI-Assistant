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

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { activeConversationId, setActiveConversationId, fetchConversations } =
    useChat();
  const justCreatedRef = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

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
          const response = await api.get(
            `/conversations/${activeConversationId}`,
          );
          const formattedMessages = response.data.messages.map(
            (m: {
              id: number;
              role: "user" | "assistant";
              content: string;
              sources: string | null;
              confidence: string | null;
            }) => ({
              id: m.id,
              role: m.role,
              content: m.content,
              sources: m.sources ? JSON.parse(m.sources) : undefined,
              confidence: m.confidence || undefined,
            }),
          );
          setMessages(formattedMessages);
        } catch (error) {
          console.error("Failed to load conversation:", error);
          toast.error("Failed to load conversation.");
        } finally {
          setLoading(false);
        }
      };

      fetchMessages();
    } else {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setMessages([]);
    }
  }, [activeConversationId]);

  const handleFeedback = async (
    messageId: number,
    rating: "up" | "down",
    index: number,
  ) => {
    try {
      await api.post("/feedback", { message_id: messageId, rating });
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[index] = { ...newMessages[index], feedback: rating };
        return newMessages;
      });
      toast.success("Feedback submitted!");
    } catch (error) {
      console.error("Failed to submit feedback:", error);
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
    } catch (error) {
      console.error("Failed to get answer:", error);
      toast.error("Failed to get answer.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I encountered an error while trying to answer your question.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Disclaimer */}
      <div className="backdrop-blur-md bg-white/40 border-b border-white/20 p-2 flex items-center justify-center gap-2 flex-shrink-0 z-10">
        <AlertTriangle className="h-4 w-4 text-amber-500 flex-shrink-0" />
        <p className="text-xs text-gray-500 font-medium">
          <span className="font-semibold text-gray-600">
            Medical Disclaimer:
          </span>{" "}
          AI can make mistakes. Always consult a qualified healthcare provider.
        </p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 pb-32">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-gray-500 animate-in fade-in zoom-in duration-500">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-3xl flex items-center justify-center shadow-inner mb-6 rotate-3">
                <Bot className="h-10 w-10 text-indigo-500 -rotate-3" />
              </div>
              <h2 className="text-2xl font-semibold text-gray-800 tracking-tight">
                Medical QA Assistant
              </h2>
              <p className="text-sm mt-3 text-center max-w-md text-gray-500 leading-relaxed">
                Ask me a medical question. I will search trusted medical
                literature to find the answer.
              </p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`flex gap-4 transition-all duration-500 ease-out animate-in fade-in slide-in-from-bottom-4`}
              >
                <div className="flex-shrink-0 mt-1">
                  {msg.role === "user" ? (
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center shadow-md shadow-blue-500/20">
                      <User className="h-4 w-4 text-white" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-full flex items-center justify-center shadow-md shadow-emerald-500/20">
                      <Bot className="h-4 w-4 text-white" />
                    </div>
                  )}
                </div>
                <div
                  className={`flex-1 space-y-4 min-w-0 overflow-hidden p-5 rounded-2xl shadow-sm border ${
                    msg.role === "user"
                      ? "bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-tl-sm border-transparent shadow-blue-500/10"
                      : "bg-white/80 backdrop-blur-md text-gray-800 rounded-tl-sm border-white/60 shadow-[0_4px_24px_rgba(0,0,0,0.02)]"
                  }`}
                >
                  <div
                    className={`prose prose-sm sm:prose max-w-none break-words ${msg.role === "user" ? "text-white prose-p:text-white" : "text-gray-800"}`}
                  >
                    {msg.role === "user" ? (
                      <p className="whitespace-pre-wrap m-0 font-medium">
                        {msg.content}
                      </p>
                    ) : (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    )}
                  </div>

                  {/* Sources & Confidence */}
                  {msg.role === "assistant" && msg.sources && (
                    <div className="pt-4 border-t border-gray-200 mt-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium uppercase tracking-wider text-gray-500">
                            Confidence
                          </span>
                          <span
                            className={`text-xs px-2 py-1 rounded-full font-medium ${
                              msg.confidence?.toLowerCase() === "high"
                                ? "bg-emerald-100 text-emerald-700"
                                : msg.confidence?.toLowerCase() === "medium"
                                  ? "bg-amber-100 text-amber-700"
                                  : "bg-red-100 text-red-700"
                            }`}
                          >
                            {msg.confidence || "Unknown"}
                          </span>
                        </div>

                        {msg.id && (
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() =>
                                handleFeedback(msg.id!, "up", index)
                              }
                              className={`p-1 rounded transition-colors ${msg.feedback === "up" ? "text-blue-600 bg-blue-50" : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"}`}
                              title="Helpful"
                            >
                              <ThumbsUp className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() =>
                                handleFeedback(msg.id!, "down", index)
                              }
                              className={`p-1 rounded transition-colors ${msg.feedback === "down" ? "text-red-600 bg-red-50" : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"}`}
                              title="Not Helpful"
                            >
                              <ThumbsDown className="h-4 w-4" />
                            </button>
                          </div>
                        )}
                      </div>

                      <details className="group">
                        <summary className="text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900 flex items-center gap-1 list-none select-none">
                          <Info className="h-4 w-4" />
                          <span className="flex-1">
                            View {msg.sources.length} sources
                          </span>
                          <span className="transition group-open:rotate-180">
                            <ChevronDown className="h-4 w-4" />
                          </span>
                        </summary>
                        <div className="mt-3 space-y-3 pl-5 border-l-2 border-gray-200">
                          {msg.sources.map((source, idx) => (
                            <div key={idx} className="text-sm">
                              <p className="font-medium text-gray-700">
                                {source.topic}
                              </p>
                              <p className="text-gray-500 text-xs mt-1 line-clamp-3">
                                {source.text}
                              </p>
                            </div>
                          ))}
                        </div>
                      </details>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="flex gap-4 p-5">
              <div className="flex-shrink-0 mt-1">
                <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-full flex items-center justify-center shadow-md shadow-emerald-500/20">
                  <Bot className="h-4 w-4 text-white" />
                </div>
              </div>
              <div className="flex-1 flex items-center h-10 bg-white/60 backdrop-blur-md rounded-2xl rounded-tl-sm px-5 border border-white/60 shadow-sm w-fit max-w-[100px]">
                <div className="flex space-x-1.5">
                  <div className="w-2 h-2 bg-emerald-500/60 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="w-2 h-2 bg-emerald-500/60 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="w-2 h-2 bg-emerald-500/60 rounded-full animate-bounce"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 p-4 sm:p-6 bg-gradient-to-t from-white/80 via-white/50 to-transparent pt-12 backdrop-blur-[2px]">
        <div className="max-w-3xl mx-auto">
          <form
            onSubmit={handleSubmit}
            className="relative flex items-end shadow-[0_8px_30px_rgba(0,0,0,0.06)] border border-white/60 bg-white/70 backdrop-blur-xl rounded-3xl overflow-hidden focus-within:ring-2 focus-within:ring-indigo-500/50 focus-within:border-indigo-500 transition-all duration-300 hover:shadow-[0_8px_30px_rgba(0,0,0,0.08)]"
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
              placeholder="Ask a medical question..."
              className="w-full max-h-48 py-4 pl-6 pr-14 bg-transparent border-0 focus:ring-0 resize-none outline-none text-sm text-gray-900 placeholder:text-gray-400 font-medium"
              rows={1}
              style={{ minHeight: "56px" }}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="absolute right-2 bottom-2 p-2.5 rounded-2xl text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:shadow-lg hover:scale-105 active:scale-95 disabled:opacity-50 disabled:hover:scale-100 disabled:shadow-none transition-all duration-200"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
          <p className="text-center text-xs text-gray-400 mt-2">
            AI can make mistakes. Verify important medical information.
          </p>
        </div>
      </div>
    </div>
  );
}
