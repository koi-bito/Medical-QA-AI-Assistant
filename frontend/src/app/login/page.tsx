"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import api from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import {
  Heart,
  Mail,
  Lock,
  ArrowRight,
  Stethoscope,
} from "lucide-react";

export default function LoginPage() {
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);

  const router = useRouter();
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);
      const response = await api.post("/auth/login", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      await login(response.data.access_token);
      router.push("/");
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } };
      setError(e.response?.data?.detail || "Invalid email or password. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex"
      style={{
        backgroundColor: "#f5f0e8",
        backgroundImage:
          "radial-gradient(ellipse 80% 60% at 10% 0%, rgba(200,16,46,0.07) 0%, transparent 60%), radial-gradient(ellipse 60% 50% at 90% 100%, rgba(74,124,89,0.07) 0%, transparent 55%)",
      }}
    >
      {/* ── Left decorative panel (hidden on mobile) ── */}
      <div
        className="hidden lg:flex flex-col justify-between w-96 flex-shrink-0 p-10"
        style={{
          background: "linear-gradient(160deg, #c8102e 0%, #6b0518 60%, #2d4e38 100%)",
        }}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
            <Heart className="h-5 w-5 text-white fill-white" />
          </div>
          <span className="text-white font-bold text-lg tracking-tight">MedQA Assistant</span>
        </div>

        <div className="space-y-6">
          <div>
            <Stethoscope className="h-10 w-10 text-white/60 mb-4" />
            <h2 className="text-3xl font-bold text-white leading-tight">
              Your AI-powered<br />Medical Companion
            </h2>
            <p className="text-white/60 text-sm mt-3 leading-relaxed">
              Get sourced, evidence-based answers to your medical questions — powered by fine-tuned AI and a 100k+ conversation knowledge base.
            </p>
          </div>
          <div className="space-y-3">
            {["Evidence-based answers", "Source citations included", "Conversation history saved"].map((f) => (
              <div key={f} className="flex items-center gap-2.5">
                <div className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center flex-shrink-0">
                  <ArrowRight className="h-3 w-3 text-white" />
                </div>
                <span className="text-white/80 text-sm">{f}</span>
              </div>
            ))}
          </div>
        </div>

        <p className="text-white/30 text-xs">
          Not a substitute for professional medical advice.
        </p>
      </div>

      {/* ── Right form panel ── */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="flex items-center gap-3 mb-8 lg:hidden">
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
                Welcome back
              </h1>
              <p className="text-sm mt-1" style={{ color: "#9b8f85" }}>
                Sign in to access your medical assistant
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
                    id="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    className="w-full pl-10 pr-4 py-3 rounded-xl text-sm outline-none transition-all duration-150"
                    style={{ backgroundColor: "#faf7f2", border: "1.5px solid #ddd6c8", color: "#2e261d" }}
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
                    id="password"
                    type="password"
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full pl-10 pr-4 py-3 rounded-xl text-sm outline-none transition-all duration-150"
                    style={{ backgroundColor: "#faf7f2", border: "1.5px solid #ddd6c8", color: "#2e261d" }}
                    onFocus={(e) => (e.target.style.borderColor = "#c8102e")}
                    onBlur={(e) => (e.target.style.borderColor = "#ddd6c8")}
                  />
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={loading}
                className="w-full py-3.5 rounded-xl text-sm font-bold text-white transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100 flex items-center justify-center gap-2 mt-2"
                style={{
                  background: "linear-gradient(135deg, #c8102e 0%, #a80d25 100%)",
                  boxShadow: "0 4px 16px rgba(200,16,46,0.30)",
                }}
              >
                {loading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                    Signing in…
                  </>
                ) : (
                  <>Log In <ArrowRight className="h-4 w-4" /></>
                )}
              </button>
            </form>

            <p className="text-center text-sm mt-6" style={{ color: "#9b8f85" }}>
              Don&apos;t have an account?{" "}
              <Link href="/register" className="font-semibold transition-colors hover:underline" style={{ color: "#c8102e" }}>
                Register here
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
