"use client";

import React from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import { useAuth } from "@/context/AuthContext";
import { PlusCircle, LogOut } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();

  return (
    <ProtectedRoute>
      <div className="flex h-screen bg-white">
        {/* Sidebar */}
        <aside className="w-72 border-r border-gray-200 bg-gray-50 flex flex-col">
          <div className="p-4">
            <button className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
              <PlusCircle className="h-4 w-4" />
              New Chat
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4">
            {/* Conversation list will go here later */}
            <p className="text-sm text-gray-500 text-center mt-4">No conversations yet</p>
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

        {/* Main Content */}
        <main className="flex-1 flex flex-col h-screen overflow-hidden bg-white">
          {children}
        </main>
      </div>
    </ProtectedRoute>
  );
}
