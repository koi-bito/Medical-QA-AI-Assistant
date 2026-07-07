"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import api from "@/lib/api";
import { Heart, Mail, Lock, User, ArrowRight } from "lucide-react";

export default function RegisterPage() {
  const [email, setEmail]       = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await api.post("/auth/register", { email, username, password });
      router.push("/login?registered=true");
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } };
      setError(e.response?.data?.detail || "Failed to create account. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "w-full pl-10 pr-4 py-3 rounded-xl text-sm outline-none transition-all duration-150";
  const inputStyle = { backgroundColor: "#faf7f2", border: "1.5px solid #ddd6c8", color: "#2e261d" };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-6"
      style={{
        backgroundColor: "#f5f0e8",
        backgroundImage:
          "radial-gradient(ellipse 80% 60% at 10% 0%, rgba(200,16,46,0.07) 0%, transparent 60%), radial-gradient(ellipse 60% 50% at 90% 100%, rgba(74,124,89,0.07) 0%, transparent 55%)",
      }}
    >
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div
            style={{ background: "linear-gradient(135deg, #c8102e, #8b0a1e)" }}
            className="w-10 h-10 rounded-xl flex items-center justify-center shadow-md"
          >
            <Heart className="h-5 w-5 text-white fill-white" />
          </div>
          <span className="font-bold text-xl" style={{ color: "#2e261d" }}>MedQA Assistant</span>
        </div>

        {/* Card */}
        <div
          className="rounded-3xl p-8 shadow-xl"
          style={{ backgroundColor: "#fffdf9", border: "1px solid #ede8df" }}
        >
          <div className="mb-8">
            <h1 className="text-2xl font-bold tracking-tight" style={{ color: "#2e261d" }}>
              Create an account
            </h1>
            <p className="text-sm mt-1" style={{ color: "#9b8f85" }}>
              Join to access your personal medical assistant
            </p>
          </div>

          {error && (
            <div
              className="mb-5 px-4 py-3 rounded-xl text-sm font-medium"
              style={{ background: "#fff1f1", border: "1px solid #ffdede", color: "#8b0a1e" }}
            >
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "#9b8f85" }}>
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: "#c8a98a" }} />
                <input
                  id="email" type="email" autoComplete="email" required
                  value={email} onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className={inputClass} style={inputStyle}
                  onFocus={(e) => (e.target.style.borderColor = "#c8102e")}
                  onBlur={(e) => (e.target.style.borderColor = "#ddd6c8")}
                />
              </div>
            </div>

            {/* Username */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "#9b8f85" }}>
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: "#c8a98a" }} />
                <input
                  id="username" type="text" autoComplete="off" required
                  value={username} onChange={(e) => setUsername(e.target.value)}
                  placeholder="johndoe"
                  className={inputClass} style={inputStyle}
                  onFocus={(e) => (e.target.style.borderColor = "#c8102e")}
                  onBlur={(e) => (e.target.style.borderColor = "#ddd6c8")}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "#9b8f85" }}>
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: "#c8a98a" }} />
                <input
                  id="password" type="password" autoComplete="new-password" required
                  value={password} onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className={inputClass} style={inputStyle}
                  onFocus={(e) => (e.target.style.borderColor = "#c8102e")}
                  onBlur={(e) => (e.target.style.borderColor = "#ddd6c8")}
                />
              </div>
            </div>

            {/* Terms note */}
            <p className="text-xs leading-relaxed" style={{ color: "#c8a98a" }}>
              By creating an account, you agree that this tool is for informational purposes only and is not a substitute for professional medical advice.
            </p>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3.5 rounded-xl text-sm font-bold text-white transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100 flex items-center justify-center gap-2"
              style={{
                background: "linear-gradient(135deg, #c8102e 0%, #a80d25 100%)",
                boxShadow: "0 4px 16px rgba(200,16,46,0.30)",
              }}
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                  Creating account…
                </>
              ) : (
                <>Create Account <ArrowRight className="h-4 w-4" /></>
              )}
            </button>
          </form>

          <p className="text-center text-sm mt-6" style={{ color: "#9b8f85" }}>
            Already have an account?{" "}
            <Link href="/login" className="font-semibold transition-colors hover:underline" style={{ color: "#c8102e" }}>
              Log in here
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
