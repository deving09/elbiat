"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { api, type LeaderboardEntry, type Model } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Trophy,
  ExternalLink,
  FileText,
  Play,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Medal,
} from "lucide-react";

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case "COMPLETE":
      return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />Complete</Badge>;
    case "RUNNING":
      return <Badge variant="warning"><Loader2 className="h-3 w-3 mr-1 animate-spin" />Running</Badge>;
    case "QUEUED":
      return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Queued</Badge>;
    case "FAILED":
      return <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" />Failed</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

function LeaderboardTable({ entries, metricKey }: { entries: LeaderboardEntry[]; metricKey: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b">
            <th className="text-left py-3 px-4 font-medium text-muted-foreground">Rank</th>
            <th className="text-left py-3 px-4 font-medium text-muted-foreground">Model</th>
            <th className="text-right py-3 px-4 font-medium text-muted-foreground">{metricKey}</th>
            <th className="text-left py-3 px-4 font-medium text-muted-foreground">Date</th>
            <th className="text-left py-3 px-4 font-medium text-muted-foreground">Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, index) => (
            <tr
              key={entry.run_id}
              className="border-b hover:bg-muted/50 transition-colors"
            >
              <td className="py-3 px-4">
                <div className="flex items-center">
                  {index === 0 && <Medal className="h-5 w-5 text-yellow-500 mr-2" />}
                  {index === 1 && <Medal className="h-5 w-5 text-gray-400 mr-2" />}
                  {index === 2 && <Medal className="h-5 w-5 text-amber-600 mr-2" />}
                  {index > 2 && <span className="w-7 text-center">{index + 1}</span>}
                </div>
              </td>
              <td className="py-3 px-4">
                <div>
                  <div className="font-medium">{entry.model_display_name}</div>
                  <div className="text-xs text-muted-foreground">{entry.model_name}</div>
                </div>
              </td>
              <td className="py-3 px-4 text-right">
                <span className="font-mono font-semibold">
                  {entry.primary_metric !== null
                    ? entry.primary_metric.toFixed(2)
                    : "—"}
                </span>
              </td>
              <td className="py-3 px-4 text-muted-foreground text-sm">
                {new Date(entry.run_date).toLocaleDateString()}
              </td>
              <td className="py-3 px-4">
                <StatusBadge status={entry.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TaskPage() {
  const params = useParams();
  const taskName = params.name as string;
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState<string>("");



  // Add state for selected metric
  const [selectedMetric, setSelectedMetric] = useState<string>("");
  
  // Add query for available metrics
  const { data: availableMetrics } = useQuery({
    queryKey: ["metrics", taskName],
    queryFn: () => api.getAvailableMetrics(taskName),
  });

  // Update leaderboard query to use selected metric
  const { data: leaderboard, isLoading: leaderboardLoading } = useQuery({
    queryKey: ["leaderboard", taskName, selectedMetric],
    queryFn: () => api.getLeaderboard(taskName, selectedMetric || undefined),
  });

  const { data: task, isLoading: taskLoading } = useQuery({
    queryKey: ["task", taskName],
    queryFn: () => api.getTask(taskName),
  });

  /*
  const { data: leaderboard, isLoading: leaderboardLoading } = useQuery({
    queryKey: ["leaderboard", taskName],
    queryFn: () => api.getLeaderboard(taskName),
  });
  */

  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: api.getModels,
  });

  const triggerMutation = useMutation({
    mutationFn: (modelName: string) => api.triggerEval(taskName, modelName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["leaderboard", taskName] });
      setSelectedModel("");
    },
  });

  if (taskLoading) {
    return (
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 w-64 bg-muted rounded" />
          <div className="h-4 w-96 bg-muted rounded" />
          <div className="h-64 bg-muted rounded-xl mt-8" />
        </div>
      </main>
    );
  }

  if (!task) {
    return (
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="text-lg font-medium">Task not found</p>
          </CardContent>
        </Card>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Task header */}
      <div className="mb-8">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <Trophy className="h-8 w-8 text-primary" />
              <h1 className="text-3xl font-bold">{task.display_name}</h1>
            </div>
            <div className="flex items-center space-x-4 text-muted-foreground">
              {task.num_examples && (
                <span>{task.num_examples.toLocaleString()} examples</span>
              )}
              {task.dataset_version && (
                <>
                  <span>•</span>
                  <span>Version: {task.dataset_version}</span>
                </>
              )}
              <span>•</span>
              <span>Metric: {task.primary_metric_key}</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {task.paper_url && (
              <a
                href={task.paper_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm">
                  <FileText className="h-4 w-4 mr-2" />
                  Paper
                  <ExternalLink className="h-3 w-3 ml-2" />
                </Button>
              </a>
            )}
          </div>
        </div>
        {task.description && (
          <p className="mt-4 text-muted-foreground max-w-3xl">
            {task.description}
          </p>
        )}
      </div>

      {/* Run new evaluation */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-lg">Run New Evaluation</CardTitle>
          <CardDescription>
            Select a model to evaluate on this benchmark
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="flex h-9 w-full max-w-xs rounded-md border border-border bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
            >
              <option value="">Select a model...</option>
              {models?.map((model) => (
                <option key={model.id} value={model.name}>
                  {model.display_name}
                </option>
              ))}
            </select>
            <Button
              onClick={() => triggerMutation.mutate(selectedModel)}
              disabled={!selectedModel || triggerMutation.isPending}
            >
              {triggerMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Run Evaluation
            </Button>
          </div>
          {triggerMutation.isError && (
            <p className="text-destructive text-sm mt-2">
              Failed to trigger evaluation. Please try again.
            </p>
          )}
          {triggerMutation.isSuccess && (
            <p className="text-green-600 text-sm mt-2">
              Evaluation queued successfully!
            </p>
          )}
        </CardContent>
      </Card>

      




      {/* Leaderboard */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <Trophy className="h-5 w-5" />
                <span>Leaderboard</span>
              </CardTitle>
              <CardDescription>
                Best results per model, sorted by {selectedMetric || task.primary_metric_key}
              </CardDescription>
            </div>
      
            {/* Metric selector */}
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="flex h-9 rounded-md border border-border bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
            >
              <option value="">Default ({task.primary_metric_key})</option>
              {availableMetrics
                ?.filter((metric) => metric !== task.primary_metric_key)
                .map((metric) => (
                  <option key={metric} value={metric}>
                    {metric}
                  </option>
                ))}
            </select>
          </div>

        </CardHeader>
        <CardContent>
          {leaderboardLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-12 bg-muted rounded animate-pulse" />
              ))}
            </div>
          ) : leaderboard && leaderboard.length > 0 ? (
            <LeaderboardTable 
              entries={leaderboard} 
              metricKey={selectedMetric || task.primary_metric_key} />
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <Trophy className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No results yet</p>
              <p className="text-sm">Run an evaluation to see results here</p>
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
