"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api, ComparisonFeedback } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import {
  Loader2,
  ThumbsUp,
  ThumbsDown,
  ChevronLeft,
  ChevronRight,
  Image as ImageIcon,
  MessageSquare,
  ArrowRight,
  Filter,
} from "lucide-react";

// Extended type with responses - backend schema needs update to return these
interface ComparisonSample extends ComparisonFeedback {
  base_response?: string;
  adapted_response?: string;
}

type FilterType = "all" | "adapted_wins" | "base_wins" | "ties";

function PreferenceBadge({ preference }: { preference: string | null }) {
  if (!preference) return null;

  const config: Record<string, { label: string; className: string }> = {
    adapted: { label: "Your model preferred", className: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" },
    base: { label: "Base preferred", className: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200" },
    tie: { label: "Tie", className: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" },
  };

  const c = config[preference];
  if (!c) return null;

  return <Badge className={c.className}>{c.label}</Badge>;
}

function ThumbsIndicator({ thumbs, label }: { thumbs: string | null; label: string }) {
  if (!thumbs) return null;
  return (
    <span className={cn(
      "inline-flex items-center gap-1 text-xs",
      thumbs === "up" ? "text-green-600" : "text-red-600"
    )}>
      {thumbs === "up" ? <ThumbsUp className="h-3 w-3" /> : <ThumbsDown className="h-3 w-3" />}
      {label}
    </span>
  );
}

function SampleCard({ sample }: { sample: ComparisonSample }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="overflow-hidden">
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-muted rounded-lg">
              <ImageIcon className="h-4 w-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-sm font-medium">Image #{sample.image_id}</p>
              <p className="text-xs text-muted-foreground">
                {new Date(sample.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
          <PreferenceBadge preference={sample.preference} />
        </div>

        {/* Prompt */}
        <div className="mb-3">
          <p className="text-xs text-muted-foreground mb-1">Prompt</p>
          <p className="text-sm bg-muted p-2 rounded">{sample.prompt}</p>
        </div>

        {/* Thumbs indicators */}
        <div className="flex gap-4 mb-3">
          <ThumbsIndicator thumbs={sample.base_thumbs} label="Base" />
          <ThumbsIndicator thumbs={sample.adapted_thumbs} label="Yours" />
        </div>

        {/* Responses (if available and expanded) */}
        {(sample.base_response || sample.adapted_response) && (
          <>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(!expanded)}
              className="w-full justify-between"
            >
              {expanded ? "Hide responses" : "Show responses"}
              {expanded ? <ChevronLeft className="h-4 w-4 rotate-90" /> : <ChevronRight className="h-4 w-4 rotate-90" />}
            </Button>

            {expanded && (
              <div className="mt-3 grid md:grid-cols-2 gap-3">
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-xs font-medium mb-1 text-muted-foreground">Base Model</p>
                  <p className="text-sm whitespace-pre-wrap">{sample.base_response}</p>
                </div>
                <div className="p-3 bg-primary/5 rounded-lg border border-primary/20">
                  <p className="text-xs font-medium mb-1 text-primary">Your Model</p>
                  <p className="text-sm whitespace-pre-wrap">{sample.adapted_response}</p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
}

export default function SamplesPage() {
  const { user } = useAuth();
  const [samples, setSamples] = useState<ComparisonSample[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<FilterType>("all");

  useEffect(() => {
    if (!user) return;

    api.getComparisonFeedback()
      .then(setSamples)
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, [user]);

  const filteredSamples = samples.filter((s) => {
    if (filter === "all") return true;
    if (filter === "adapted_wins") return s.preference === "adapted";
    if (filter === "base_wins") return s.preference === "base";
    if (filter === "ties") return s.preference === "tie";
    return true;
  });

  const counts = {
    all: samples.length,
    adapted_wins: samples.filter((s) => s.preference === "adapted").length,
    base_wins: samples.filter((s) => s.preference === "base").length,
    ties: samples.filter((s) => s.preference === "tie").length,
  };

  if (isLoading) {
    return (
      <main className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </main>
    );
  }

  return (
    <main className="p-4">
      <div className="mx-auto max-w-4xl space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold">Inference Samples</h1>
            <p className="text-muted-foreground">
              Browse past comparisons to see how your model differs
            </p>
          </div>
          <Link href="/compare">
            <Button>
              New Comparison
              <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </Link>
        </div>

        {/* Filters */}
        {samples.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <Filter className="h-4 w-4 text-muted-foreground" />
            {(["all", "adapted_wins", "base_wins", "ties"] as FilterType[]).map((f) => {
              const labels: Record<FilterType, string> = {
                all: "All",
                adapted_wins: "Your Model Wins",
                base_wins: "Base Wins",
                ties: "Ties",
              };
              return (
                <Button
                  key={f}
                  variant={filter === f ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFilter(f)}
                >
                  {labels[f]} ({counts[f]})
                </Button>
              );
            })}
          </div>
        )}

        {/* Samples list */}
        {samples.length === 0 ? (
          <Card className="p-8 text-center">
            <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="font-semibold mb-2">No comparison samples yet</h3>
            <p className="text-muted-foreground mb-4">
              Run side-by-side comparisons to build your sample library
            </p>
            <Link href="/compare">
              <Button>Start Comparing</Button>
            </Link>
          </Card>
        ) : filteredSamples.length === 0 ? (
          <Card className="p-8 text-center">
            <p className="text-muted-foreground">No samples match this filter</p>
            <Button variant="ghost" onClick={() => setFilter("all")} className="mt-2">
              Show all
            </Button>
          </Card>
        ) : (
          <div className="space-y-4">
            {filteredSamples.map((sample) => (
              <SampleCard key={sample.id} sample={sample} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}