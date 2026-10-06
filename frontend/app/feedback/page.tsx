"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  MessageSquare,
  X,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  Star,
  Filter,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getAccessToken } from "@/lib/api";

interface AttributionScore {
  benchmark: string;
  method: string;
  checkpoint_id: string;
  influence_score: number;
  z_score: number | null;
  rank: number | null;
}

interface Convo {
  id: number;
  image_id: number;
  feedback: string;
  enabled: boolean;
  created_at: string;
  prompt: string;
  response: string;
  feedback_length: number;
  attributions: AttributionScore[];
  model_name: string;
  task: string;
}

interface FeedbackListResponse {
  items: Convo[];
  total: number;
  page: number;
  page_size: number;
}

interface AttributionFilters {
  benchmarks: string[];
  checkpoints: string[];
  methods: string[];
}

const fetchFilters = async (): Promise<AttributionFilters> => {
  const token = getAccessToken();
  const response = await fetch("/api/feedback/attribution-filters", {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) return { benchmarks: [], checkpoints: [], methods: [] };
  return response.json();
};

const fetchFeedback = async (
  page: number,
  pageSize: number,
  sortBy: string,
  sortOrder: string,
  benchmark?: string,
  checkpointId?: string,
  method?: string
): Promise<FeedbackListResponse> => {
  const token = getAccessToken();
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
    sort_by: sortBy,
    sort_order: sortOrder,
  });
  if (benchmark) params.set("benchmark", benchmark);
  if (checkpointId) params.set("checkpoint_id", checkpointId);
  if (method) params.set("method", method);

  const response = await fetch(`/api/feedback?${params}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error("Failed to fetch feedback");
  return response.json();
};

// Helper to get display score from attributions
const getDisplayScore = (attributions: AttributionScore[], benchmark?: string): number | null => {
  if (!attributions.length) return null;
  if (benchmark) {
    const match = attributions.find((a) => a.benchmark === benchmark);
    return match?.influence_score ?? null;
  }
  return attributions[0]?.influence_score ?? null;
};

const formatScore = (score: number | null): string => {
  if (score === null) return "—";
  return score.toFixed(4);
};

const getScoreColor = (score: number | null): string => {
  if (score === null) return "bg-gray-500";
  if (score > 0.03) return "bg-green-500";
  if (score > 0) return "bg-green-400";
  if (score > -0.01) return "bg-yellow-500";
  return "bg-red-500";
};

export default function FeedbackPage() {
  const router = useRouter();
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
  const [sortBy, setSortBy] = useState<string>("created_at");
  const [sortOrder, setSortOrder] = useState<string>("desc");

  // Attribution filters
  const [benchmark, setBenchmark] = useState<string | undefined>();
  const [checkpointId, setCheckpointId] = useState<string | undefined>();
  const [method, setMethod] = useState<string | undefined>();

  // Fetch available filters
  const { data: filters } = useQuery({
    queryKey: ["attribution-filters"],
    queryFn: fetchFilters,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["feedback", page, pageSize, sortBy, sortOrder, benchmark, checkpointId, method],
    queryFn: () => fetchFeedback(page, pageSize, sortBy, sortOrder, benchmark, checkpointId, method),
  });

  const openDetail = (convo: Convo) => {
    // Build return URL with current filters
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("sort_by", sortBy);
    params.set("sort_order", sortOrder);
    if (benchmark) params.set("benchmark", benchmark);
    if (checkpointId) params.set("checkpoint_id", checkpointId);
    if (method) params.set("method", method);

    router.push(`/feedback/${convo.id}?from=${encodeURIComponent(`/feedback?${params.toString()}`)}`);
  };

  const toggleSort = () => {
    setSortOrder(sortOrder === "desc" ? "asc" : "desc");
    setPage(1);
  };

  const clearFilters = () => {
    setBenchmark(undefined);
    setCheckpointId(undefined);
    setMethod(undefined);
    setPage(1);
  };

  const hasFilters = benchmark || checkpointId || method;
  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Feedback</h1>
          <p className="text-muted-foreground mt-1">
            View and manage your feedback contributions
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-6 p-4 bg-muted/50 rounded-lg">
        <Filter className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Filters:</span>

        <Select value={benchmark ?? "all"} onValueChange={(v) => { setBenchmark(v === "all" ? undefined : v); setPage(1); }}>
          <SelectTrigger className="w-36">
            <SelectValue placeholder="Benchmark" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Benchmarks</SelectItem>
            {filters?.benchmarks.map((b) => (
              <SelectItem key={b} value={b}>{b}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={checkpointId ?? "all"} onValueChange={(v) => { setCheckpointId(v === "all" ? undefined : v); setPage(1); }}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Checkpoint" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Checkpoints</SelectItem>
            {filters?.checkpoints.map((c) => (
              <SelectItem key={c} value={c}>{c}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={method ?? "all"} onValueChange={(v) => { setMethod(v === "all" ? undefined : v); setPage(1); }}>
          <SelectTrigger className="w-32">
            <SelectValue placeholder="Method" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Methods</SelectItem>
            {filters?.methods.map((m) => (
              <SelectItem key={m} value={m}>{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {hasFilters && (
          <Button variant="ghost" size="sm" onClick={clearFilters}>
            <X className="h-4 w-4 mr-1" /> Clear
          </Button>
        )}

        <div className="flex-1" />

        <span className="text-sm text-muted-foreground">Sort:</span>
        <Select value={sortBy} onValueChange={(v) => { setSortBy(v); setPage(1); }}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="created_at">Date</SelectItem>
            <SelectItem value="feedback_length">Feedback Length</SelectItem>
            <SelectItem value="attribution_score" disabled={!benchmark || !checkpointId}>
              Attribution Score
            </SelectItem>
          </SelectContent>
        </Select>
        <Button variant="outline" size="icon" onClick={toggleSort}>
          <ArrowUpDown className={cn("h-4 w-4 transition-transform", sortOrder === "asc" && "rotate-180")} />
        </Button>
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="aspect-square bg-muted rounded-xl animate-pulse" />
          ))}
        </div>
      ) : data && data.items.length > 0 ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {data.items.map((convo) => {
              const score = getDisplayScore(convo.attributions, benchmark);
              return (
                <Card
                  key={convo.id}
                  className={cn(
                    "group overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary transition-all",
                    !convo.enabled && "opacity-50"
                  )}
                  onClick={() => openDetail(convo)}
                >
                  <div className="relative aspect-square">
                    <img
                     src={`/images/${convo.image_id}/file`}
                      alt={`Image ${convo.image_id}`}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent flex flex-col justify-end p-3">
                      <p className="text-white text-sm line-clamp-2">{convo.prompt}</p>
                    </div>
                    <div className="absolute top-2 left-2">
                      <Badge className={cn("text-xs text-white", getScoreColor(score))}>
                        <Star className="h-3 w-3 mr-1" />
                        {formatScore(score)}
                      </Badge>
                    </div>
                    <div className="absolute top-2 right-2">
                      <Badge variant={convo.enabled ? "default" : "secondary"} className="text-xs">
                        {convo.enabled ? "Enabled" : "Disabled"}
                      </Badge>
                    </div>
                  </div>
                  <CardContent className="p-3">
                    <p className="text-xs text-muted-foreground line-clamp-1">{convo.feedback}</p>
                    <div className="flex items-center justify-between mt-1">
                      <p className="text-xs text-muted-foreground">
                        {new Date(convo.created_at).toLocaleDateString()}
                      </p>
                      {convo.attributions.length > 0 && (
                        <p className="text-xs text-muted-foreground">
                          {convo.attributions.length} score{convo.attributions.length > 1 ? "s" : ""}
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <div className="flex items-center justify-center gap-4 mt-8">
            <Button
              variant="outline"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft className="h-4 w-4 mr-1" /> Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              Page {page} of {totalPages} ({data.total} total)
            </span>
            <Button
              variant="outline"
              onClick={() => setPage((p) => p + 1)}
              disabled={page >= totalPages}
            >
              Next <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <MessageSquare className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">No feedback yet</p>
            <p className="text-muted-foreground">Start chatting and provide feedback to see it here</p>
          </CardContent>
        </Card>
      )}
    </main>
  );
}