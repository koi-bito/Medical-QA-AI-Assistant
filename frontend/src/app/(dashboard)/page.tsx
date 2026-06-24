"use client";

import React, { useState, useRef, useEffect } from "react";
import api from "@/lib/api";
import ReactMarkdown from "react-markdown";
import { Send, User, Bot, AlertTriangle, Info, ChevronDown } from "lucide-react";

interface Source {
  chunk_id: string;
  topic: string;
  text: string;
  distance: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  confidence?: string;
}

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

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
        // conversation_id will be added in Day 56
      });
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.data.answer,
          sources: response.data.sources,
          confidence: response.data.confidence,
        },
      ]);
    } catch (error) {
      console.error("Failed to get answer:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error while trying to answer your question.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Disclaimer */}
      <div className="bg-amber-50 border-b border-amber-200 p-3 flex items-start gap-3 flex-shrink-0 z-10">
        <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-amber-800">
          <strong>Medical Disclaimer:</strong> This AI assistant provides general medical information from trusted sources, not professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.
        </p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 pb-32">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-gray-500">
              <Bot className="h-12 w-12 text-gray-300 mb-4" />
              <h2 className="text-lg font-medium text-gray-700">Medical QA Assistant</h2>
              <p className="text-sm mt-2 text-center max-w-md">
                Ask me a medical question. I will search trusted medical literature to find the answer.
              </p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`flex gap-4 ${msg.role === "assistant" ? "bg-gray-50 rounded-xl p-4 sm:p-6" : "px-4 sm:px-6"}`}
              >
                <div className="flex-shrink-0 mt-1">
                  {msg.role === "user" ? (
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                      <User className="h-5 w-5 text-white" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-emerald-600 rounded-full flex items-center justify-center">
                      <Bot className="h-5 w-5 text-white" />
                    </div>
                  )}
                </div>
                <div className="flex-1 space-y-4 min-w-0 overflow-hidden">
                  <div className={`prose prose-sm sm:prose max-w-none text-gray-800 break-words`}>
                    {msg.role === "user" ? (
                      <p className="whitespace-pre-wrap m-0">{msg.content}</p>
                    ) : (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    )}
                  </div>
                  
                  {/* Sources & Confidence */}
                  {msg.role === "assistant" && msg.sources && (
                    <div className="pt-4 border-t border-gray-200 mt-4 space-y-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium uppercase tracking-wider text-gray-500">
                          Confidence
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                          msg.confidence?.toLowerCase() === 'high' ? 'bg-emerald-100 text-emerald-700' :
                          msg.confidence?.toLowerCase() === 'medium' ? 'bg-amber-100 text-amber-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {msg.confidence || "Unknown"}
                        </span>
                      </div>
                      
                      <details className="group">
                        <summary className="text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900 flex items-center gap-1 list-none select-none">
                          <Info className="h-4 w-4" />
                          <span className="flex-1">View {msg.sources.length} sources</span>
                          <span className="transition group-open:rotate-180">
                            <ChevronDown className="h-4 w-4" />
                          </span>
                        </summary>
                        <div className="mt-3 space-y-3 pl-5 border-l-2 border-gray-200">
                          {msg.sources.map((source, idx) => (
                            <div key={idx} className="text-sm">
                              <p className="font-medium text-gray-700">{source.topic}</p>
                              <p className="text-gray-500 text-xs mt-1 line-clamp-3">{source.text}</p>
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
            <div className="flex gap-4 bg-gray-50 rounded-xl p-4 sm:p-6">
              <div className="flex-shrink-0 mt-1">
                <div className="w-8 h-8 bg-emerald-600 rounded-full flex items-center justify-center">
                  <Bot className="h-5 w-5 text-white" />
                </div>
              </div>
              <div className="flex-1 flex items-center h-8">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-white via-white to-transparent pt-10">
        <div className="max-w-3xl mx-auto">
          <form
            onSubmit={handleSubmit}
            className="relative flex items-end shadow-sm border border-gray-300 bg-white rounded-2xl overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition-shadow"
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
              className="w-full max-h-48 py-3 pl-4 pr-12 bg-transparent border-0 focus:ring-0 resize-none outline-none text-sm text-gray-900 placeholder:text-gray-500"
              rows={1}
              style={{ minHeight: '52px' }}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="absolute right-2 bottom-2 p-2 rounded-xl text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:bg-gray-300 disabled:text-gray-500 transition-colors"
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
