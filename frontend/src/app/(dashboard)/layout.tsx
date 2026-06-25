"use client";

import React from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import Sidebar from "@/components/Sidebar";
import { ChatProvider } from "@/context/ChatContext";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <ProtectedRoute>
      <ChatProvider>
        <div className="flex h-screen bg-white">
          <Sidebar />
          {/* Main Content */}
          <main className="flex-1 flex flex-col h-screen overflow-hidden bg-white">
            {children}
          </main>
        </div>
      </ChatProvider>
    </ProtectedRoute>
  );
}
