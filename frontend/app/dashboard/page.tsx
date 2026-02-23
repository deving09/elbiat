"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MessageSquare, Trophy, Image as ImageIcon, ArrowRight, Activity } from "lucide-react";

export default function DashboardPage() {
  const { user } = useAuth();

  const { data: tasks } = useQuery({
    queryKey: ["tasks"],
    queryFn: api.getTasks,
  });

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Welcome section */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold">
          Welcome back, {user?.username || "there"}!
        </h1>
        <p className="text-muted-foreground mt-1">
          Here&apos;s what&apos;s happening with your VLM evaluations
        </p>
      </div>

      {/* Quick actions */}
      <div className="grid gap-4 md:grid-cols-3 mb-8">
        <Link href="/chat">
          <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Start Chat</CardTitle>
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">
                Chat with VLMs about images and documents
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/tasks">
          <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">View Tasks</CardTitle>
              <Trophy className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">
                {tasks?.length || 0} benchmarks available
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/gallery">
          <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Gallery</CardTitle>
              <ImageIcon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">
                Browse and upload images
              </p>
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* Tasks overview */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Evaluation Tasks</h2>
          <Link href="/tasks">
            <Button variant="ghost" size="sm">
              View all
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {tasks?.slice(0, 6).map((task) => (
            <Link key={task.id} href={`/tasks/${task.name}`}>
              <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{task.display_name}</CardTitle>
                      <CardDescription className="mt-1">
                        {task.num_examples ? `${task.num_examples.toLocaleString()} examples` : ""}
                      </CardDescription>
                    </div>
                    {task.run_count !== undefined && task.run_count > 0 && (
                      <Badge variant="secondary">
                        {task.run_count} runs
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {task.description || "No description available"}
                  </p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>

        {(!tasks || tasks.length === 0) && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Activity className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No tasks available yet</p>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
