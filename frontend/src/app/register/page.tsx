"use client";

import React, { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import api from "@/lib/api";
import {
  Heart,
  Mail,
  Lock,
  User,
  ArrowRight,
  ShieldCheck,
  RefreshCw,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────
type Stage = "register" | "verify";

export default function RegisterPage() {
  // ── Registration form state ─────────────────────────────────────────────────
  const [email, setEmail]       = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  // ── OTP verification state ──────────────────────────────────────────────────
  const [stage, setStage]           = useState<Stage>("register");
  const [otp, setOtp]               = useState(["", "", "", "", "", ""]);
  const otpRefs                     = useRef<(HTMLInputElement | null)[]>([]);

  // ── Shared state ────────────────────────────────────────────────────────────
  const [error, setError]     = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  // ── Styles ──────────────────────────────────────────────────────────────────
  const inputClass = "w-full pl-10 pr-4 py-3 rounded-xl text-sm outline-none transition-all duration-150";
  const inputStyle = { backgroundColor: "#faf7f2", border: "1.5px solid #ddd6c8", color: "#2e261d" };

  // ─── Register handler ────────────────────────────────────────────────────────
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await api.post("/auth/register", { email, username, password });
      setStage("verify");
      setSuccess(`A 6-digit code was sent to ${email}. It expires in 10 minutes.`);
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } };
      setError(e.response?.data?.detail || "Failed to create account. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // ─── OTP input helpers ────────────────────────────────────────────────────────
  const handleOtpChange = (index: number, value: string) => {
    if (!/^\d*$/.test(value)) return; // digits only
    const next = [...otp];
    next[index] = value.slice(-1); // keep last digit if pasting
    setOtp(next);
    if (value && index < 5) otpRefs.current[index + 1]?.focus();
  };

  const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      otpRefs.current[index - 1]?.focus();
    }
  };

  const handleOtpPaste = (e: React.ClipboardEvent) => {
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (pasted.length === 6) {
      setOtp(pasted.split(""));
      otpRefs.current[5]?.focus();
    }
  };

  // ─── Verify OTP handler ───────────────────────────────────────────────────────
  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    const code = otp.join("");
    if (code.length < 6) {
      setError("Please enter all 6 digits.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await api.post("/auth/verify-email", { email, otp: code });
      setSuccess(res.data.message || "Email verified! Redirecting to login…");
      setTimeout(() => router.push("/login?verified=true"), 1500);
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } };
      setError(e.response?.data?.detail || "Verification failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // ─── Resend OTP handler ───────────────────────────────────────────────────────
  const handleResend = async () => {
    setError("");
    setSuccess("");
    setLoading(true);
    try {
      await api.post("/auth/resend-otp", { email, password });
      setOtp(["", "", "", "", "", ""]);
      setSuccess("A new code has been sent to your email.");
      otpRefs.current[0]?.focus();
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } };
      setError(e.response?.data?.detail || "Couldn't resend code. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // ─── Shared card wrapper ──────────────────────────────────────────────────────
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
          {/* ── STAGE: Register ──────────────────────────────────────────── */}
          {stage === "register" && (
            <>
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

              <form onSubmit={handleRegister} className="space-y-5">
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

                <p className="text-xs leading-relaxed" style={{ color: "#c8a98a" }}>
                  By creating an account, you agree that this tool is for informational purposes only and is not a substitute for professional medical advice.
                </p>

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
            </>
          )}

          {/* ── STAGE: Verify OTP ────────────────────────────────────────── */}
          {stage === "verify" && (
            <>
              {/* Icon */}
              <div className="flex justify-center mb-6">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center"
                  style={{ background: "linear-gradient(135deg, #c8102e 0%, #6b0518 100%)", boxShadow: "0 8px 24px rgba(200,16,46,0.25)" }}
                >
                  <ShieldCheck className="h-8 w-8 text-white" />
                </div>
              </div>

              <div className="mb-6 text-center">
                <h1 className="text-2xl font-bold tracking-tight" style={{ color: "#2e261d" }}>
                  Verify your email
                </h1>
                <p className="text-sm mt-2 leading-relaxed" style={{ color: "#9b8f85" }}>
                  We sent a 6-digit code to<br />
                  <span className="font-semibold" style={{ color: "#2e261d" }}>{email}</span>
                </p>
              </div>

              {success && (
                <div
                  className="mb-5 px-4 py-3 rounded-xl text-sm font-medium"
                  style={{ background: "#f0fff4", border: "1px solid #b2f5c8", color: "#166534" }}
                >
                  {success}
                </div>
              )}
              {error && (
                <div
                  className="mb-5 px-4 py-3 rounded-xl text-sm font-medium"
                  style={{ background: "#fff1f1", border: "1px solid #ffdede", color: "#8b0a1e" }}
                >
                  {error}
                </div>
              )}

              <form onSubmit={handleVerify}>
                {/* OTP inputs */}
                <div className="flex gap-2 justify-center mb-6" onPaste={handleOtpPaste}>
                  {otp.map((digit, i) => (
                    <input
                      key={i}
                      ref={(el) => { otpRefs.current[i] = el; }}
                      id={`otp-${i}`}
                      type="text"
                      inputMode="numeric"
                      maxLength={1}
                      value={digit}
                      onChange={(e) => handleOtpChange(i, e.target.value)}
                      onKeyDown={(e) => handleOtpKeyDown(i, e)}
                      className="w-12 h-14 text-center text-xl font-bold rounded-xl outline-none transition-all duration-150"
                      style={{
                        backgroundColor: "#faf7f2",
                        border: `1.5px solid ${digit ? "#c8102e" : "#ddd6c8"}`,
                        color: "#2e261d",
                        boxShadow: digit ? "0 0 0 3px rgba(200,16,46,0.08)" : "none",
                      }}
                      onFocus={(e) => (e.target.style.borderColor = "#c8102e")}
                      onBlur={(e) => (e.target.style.borderColor = digit ? "#c8102e" : "#ddd6c8")}
                    />
                  ))}
                </div>

                <button
                  type="submit"
                  disabled={loading || otp.join("").length < 6}
                  className="w-full py-3.5 rounded-xl text-sm font-bold text-white transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100 flex items-center justify-center gap-2"
                  style={{
                    background: "linear-gradient(135deg, #c8102e 0%, #a80d25 100%)",
                    boxShadow: "0 4px 16px rgba(200,16,46,0.30)",
                  }}
                >
                  {loading ? (
                    <>
                      <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                      Verifying…
                    </>
                  ) : (
                    <>Verify Email <ShieldCheck className="h-4 w-4" /></>
                  )}
                </button>
              </form>

              <div className="mt-5 text-center">
                <button
                  onClick={handleResend}
                  disabled={loading}
                  className="inline-flex items-center gap-1.5 text-sm font-medium transition-colors hover:underline disabled:opacity-50"
                  style={{ color: "#c8102e" }}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Resend code
                </button>
              </div>

              <p className="text-center text-xs mt-4" style={{ color: "#c8a98a" }}>
                Wrong email?{" "}
                <button
                  onClick={() => { setStage("register"); setError(""); setSuccess(""); }}
                  className="font-semibold hover:underline"
                  style={{ color: "#c8102e" }}
                >
                  Go back
                </button>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
