"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MessageSquare, Trophy, Image as ImageIcon, ArrowRight, Activity } from "lucide-react";

export default function LandingPage() {
  return (
    <main className="min-h-screen flex items-center justify-center px-6 py-16">
      <div className="max-w-2xl text-center space-y-6">
        <h1 className="text-4xl font-bold tracking-tight">
          Collaborative Learning for AI
        </h1>

        <p className="text-lg text-muted-foreground">
          This is a project focused on collaborative learning for AI systems.
        </p>

        <p className="text-lg text-muted-foreground">
          We believe that by working together we can improve the capabilities of AI
          systems and ensure that everyone involved stands to benefit from these systems.
        </p>

        <p className="text-lg text-muted-foreground">
          There’s a lot we still need to figure out. At this stage, we sincerely ask
          that everyone who engages in the collaborative learning process makes good-faith
          efforts and provides honest feedback.
        </p>

        <div className="flex justify-center gap-3 pt-4">
          <Link href="/chat">
            <Button>Start Chat</Button>
          </Link>
          <Link href="/gallery">
            <Button variant="outline">Gallery</Button>
          </Link>
          <Link href="/dashboard">
            <Button variant="ghost">Dashboard</Button>
          </Link>
        </div>
      </div>
    </main>
  );
}