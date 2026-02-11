"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, FileText, Trophy } from "lucide-react";

export default function TasksPage() {
  const { data: tasks, isLoading } = useQuery({
    queryKey: ["tasks"],
    queryFn: api.getTasks,
  });

  if (isLoading) {
    return (
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 w-48 bg-muted rounded" />
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-48 bg-muted rounded-xl" />
            ))}
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Evaluation Tasks</h1>
        <p className="text-muted-foreground mt-1">
          Browse and run VLM benchmarks
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {tasks?.map((task) => (
          <Card key={task.id} className="flex flex-col">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2">
                  <Trophy className="h-5 w-5 text-primary" />
                  <CardTitle>{task.display_name}</CardTitle>
                </div>
              </div>
              <CardDescription className="flex items-center space-x-2">
                {task.num_examples && (
                  <span>{task.num_examples.toLocaleString()} examples</span>
                )}
                {task.dataset_version && (
                  <>
                    <span>•</span>
                    <span>{task.dataset_version}</span>
                  </>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <p className="text-sm text-muted-foreground flex-1 line-clamp-3 mb-4">
                {task.description || "No description available"}
              </p>
              
              <div className="flex items-center justify-between mt-auto pt-4 border-t">
                <div className="flex items-center space-x-2">
                  {task.run_count !== undefined && task.run_count > 0 && (
                    <Badge variant="secondary">
                      {task.run_count} runs
                    </Badge>
                  )}
                  <Badge variant="outline">
                    {task.primary_metric_key}
                  </Badge>
                </div>
                <div className="flex items-center space-x-2">
                  {task.paper_url && (
                    <a
                      href={task.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <FileText className="h-4 w-4" />
                    </a>
                  )}
                  <Link href={`/tasks/${task.name}`}>
                    <Button size="sm">
                      View
                      <ExternalLink className="ml-2 h-3 w-3" />
                    </Button>
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {(!tasks || tasks.length === 0) && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Trophy className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">No tasks available</p>
            <p className="text-muted-foreground">Check back later for new benchmarks</p>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
