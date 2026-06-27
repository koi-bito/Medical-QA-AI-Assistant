"use client";

import React, { useState } from "react";
import ProtectedRoute from "@/components/ProtectedRoute";
import Sidebar from "@/components/Sidebar";
import { ChatProvider } from "@/context/ChatContext";
import { Toaster } from "react-hot-toast";
import { Menu, X } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <ProtectedRoute>
      <ChatProvider>
        <div className="flex h-screen bg-transparent relative text-gray-800">
          {/* Mobile sidebar overlay */}
          {sidebarOpen && (
            <div 
              className="fixed inset-0 bg-black/40 backdrop-blur-sm z-20 md:hidden transition-all duration-300" 
              onClick={() => setSidebarOpen(false)}
            />
          )}
          
          <div className={`fixed inset-y-0 left-0 z-30 transform ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:relative md:translate-x-0 transition duration-200 ease-in-out md:flex`}>
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </div>

          {/* Main Content */}
          <main className="flex-1 flex flex-col h-screen overflow-hidden bg-transparent relative">
            <button
              onClick={() => setSidebarOpen(true)}
              className="md:hidden absolute top-3 left-3 z-10 p-2 rounded-md text-gray-500 hover:text-gray-900 hover:bg-gray-100"
            >
              <Menu className="h-6 w-6" />
            </button>
            {children}
            <Toaster position="bottom-right" />
          </main>
        </div>
      </ChatProvider>
    </ProtectedRoute>
  );
}
