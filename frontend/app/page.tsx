"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export default function Home() {
  const router = useRouter();
  const { isAuthenticated, isLoading, login } = useAuth();
  const [modal, setModal] = useState<"signin" | "request" | "contact" | null>(null);
  
  // Sign-in form state
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  
  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, isLoading, router]);

  // Show nothing while checking auth (prevents flash)
  if (isLoading || isAuthenticated) {
    return null;
  }

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);

    try {
      await login(email, password);
      // useEffect handles redirect when isAuthenticated becomes true
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid email or password");
    } finally {
      setSubmitting(false);
    }
  };

  // Reset form when modal closes
  const handleModalChange = (open: boolean) => {
    if (!open) {
      setModal(null);
      setEmail("");
      setPassword("");
      setError("");
    }
  };

  return (
    <main className="min-h-screen bg-black flex flex-col items-center justify-center relative">
      {/* Navigation */}
      <nav className="absolute top-8 right-8 flex flex-col items-end gap-4">
        <button
          onClick={() => setModal("signin")}
          className="text-[#1689F8] hover:text-blue-400 font-medium transition-colors"
        >
          Sign In
        </button>
        <button
          onClick={() => setModal("request")}
          className="text-[#1689F8] hover:text-blue-400 font-medium transition-colors text-right"
        >
          Request<br />Access
        </button>
        <button
          onClick={() => setModal("contact")}
          className="text-[#1689F8] hover:text-blue-400 font-medium transition-colors"
        >
          Contact Team
        </button>
      </nav>

      {/* Logo section */}
      <div className="flex items-center gap-0 -mb-10">
        <img 
          src="/logo.svg" 
          alt="Taible" 
          className="h-36 md:h-48 w-auto -mr-8"
        />
        <h1 className="text-6xl md:text-7xl tracking-wide leading-none translate-y-3" style={{ fontFamily: "'Black Ops One', cursive" }}>
          <span style={{ color: '#1689F8' }}>T</span>
          <span className="text-white">ai</span>
          <span style={{ color: '#1689F8' }}>ble</span>
        </h1>
      </div>

      {/* Tagline */}
      <p className="text-white text-xl md:text-2xl font-light">
        Building a collaborative AI future
      </p>

      {/* Sign In Modal */}
      <Dialog open={modal === "signin"} onOpenChange={handleModalChange}>
        <DialogContent className="bg-zinc-900 border-zinc-800">
          <DialogHeader>
            <DialogTitle className="text-white">Sign In</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSignIn} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Email</label>
              <Input
                type="email"
                placeholder="you@example.com"
                className="bg-zinc-950 border-zinc-800 text-white"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Password</label>
              <Input
                type="password"
                placeholder="••••••••"
                className="bg-zinc-950 border-zinc-800 text-white"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error && (
              <p className="text-red-500 text-sm">{error}</p>
            )}
            <Button 
              type="submit" 
              className="w-full bg-blue-600 hover:bg-blue-500"
              disabled={submitting}
            >
              {submitting ? "Signing in..." : "Sign In"}
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      {/* Request Access Modal */}
      <Dialog open={modal === "request"} onOpenChange={() => setModal(null)}>
        <DialogContent className="bg-zinc-900 border-zinc-800">
          <DialogHeader>
            <DialogTitle className="text-white">Request Access</DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); setModal(null); }} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Name</label>
              <Input placeholder="Your name" className="bg-zinc-950 border-zinc-800 text-white" required />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Email</label>
              <Input type="email" placeholder="you@example.com" className="bg-zinc-950 border-zinc-800 text-white" required />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Organization</label>
              <Input placeholder="Your organization" className="bg-zinc-950 border-zinc-800 text-white" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Why do you want access?</label>
              <Textarea placeholder="Tell us about your use case..." className="bg-zinc-950 border-zinc-800 text-white" />
            </div>
            <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-500">
              Submit Request
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      {/* Contact Modal */}
      <Dialog open={modal === "contact"} onOpenChange={() => setModal(null)}>
        <DialogContent className="bg-zinc-900 border-zinc-800">
          <DialogHeader>
            <DialogTitle className="text-white">Contact Team</DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); setModal(null); }} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Name</label>
              <Input placeholder="Your name" className="bg-zinc-950 border-zinc-800 text-white" required />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Email</label>
              <Input type="email" placeholder="you@example.com" className="bg-zinc-950 border-zinc-800 text-white" required />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Message</label>
              <Textarea placeholder="How can we help?" className="bg-zinc-950 border-zinc-800 text-white" required />
            </div>
            <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-500">
              Send Message
            </Button>
          </form>
        </DialogContent>
      </Dialog>
    </main>
  );
}