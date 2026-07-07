"use client";

import React, { useState } from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import Sidebar from "@/components/Sidebar";
import { ChatProvider } from "@/context/ChatContext";
import { Toaster } from "react-hot-toast";
import { Menu } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <ProtectedRoute>
      <ChatProvider>
        {/* ── Root shell — solid warm background, NO transparency holes ── */}
        <div
          style={{ backgroundColor: "#f5f0e8" }}
          className="flex h-screen overflow-hidden relative text-[#2e261d]"
        >
          {/* Mobile sidebar backdrop */}
          {sidebarOpen && (
            <div
              className="fixed inset-0 z-20 md:hidden"
              style={{ backgroundColor: "rgba(46,38,29,0.45)", backdropFilter: "blur(4px)" }}
              onClick={() => setSidebarOpen(false)}
            />
          )}

          {/* Sidebar — slides in on mobile, always visible on md+ */}
          <div
            className={`fixed inset-y-0 left-0 z-30 transform ${
              sidebarOpen ? "translate-x-0" : "-translate-x-full"
            } md:relative md:translate-x-0 transition duration-200 ease-in-out md:flex`}
          >
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </div>

          {/* Main content area */}
          <main
            style={{ backgroundColor: "#faf7f2" }}
            className="flex-1 flex flex-col h-screen overflow-hidden relative"
          >
            {/* Mobile hamburger */}
            <button
              onClick={() => setSidebarOpen(true)}
              style={{ color: "#5c5045" }}
              className="md:hidden absolute top-3 left-3 z-10 p-2 rounded-lg hover:bg-[#ede8df] transition-colors"
            >
              <Menu className="h-5 w-5" />
            </button>

            {children}

            <Toaster
              position="bottom-right"
              toastOptions={{
                style: {
                  background: "#fffdf9",
                  border: "1px solid #ede8df",
                  color: "#2e261d",
                  borderRadius: "12px",
                  fontSize: "14px",
                },
                success: {
                  iconTheme: { primary: "#4a7c59", secondary: "#f0f7ee" },
                },
                error: {
                  iconTheme: { primary: "#c8102e", secondary: "#fff1f1" },
                },
              }}
            />
          </main>
        </div>
      </ChatProvider>
    </ProtectedRoute>
  );
}
