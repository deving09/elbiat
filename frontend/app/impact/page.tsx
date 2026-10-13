"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api, UserAdaptation, ComparisonFeedback } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import {
  TrendingUp,
  Layers,
  Target,
  Award,
  ArrowRight,
  Loader2,
  BarChart3,
  Users,
  Sparkles,
} from "lucide-react";

interface AggregateStats {
  totalComparisons: number;
  adaptedPreferred: number;
  basePreferred: number;
  ties: number;
  adaptedThumbsUp: number;
  adaptedThumbsDown: number;
  baseThumbsUp: number;
  baseThumbsDown: number;
  editsProvided: number;
}

function computeStats(feedback: ComparisonFeedback[]): AggregateStats {
  return feedback.reduce(
    (acc, f) => {
      acc.totalComparisons++;
      if (f.preference === "adapted") acc.adaptedPreferred++;
      else if (f.preference === "base") acc.basePreferred++;
      else if (f.preference === "tie") acc.ties++;

      if (f.adapted_thumbs === "up") acc.adaptedThumbsUp++;
      if (f.adapted_thumbs === "down") acc.adaptedThumbsDown++;
      if (f.base_thumbs === "up") acc.baseThumbsUp++;
      if (f.base_thumbs === "down") acc.baseThumbsDown++;

      return acc;
    },
    {
      totalComparisons: 0,
      adaptedPreferred: 0,
      basePreferred: 0,
      ties: 0,
      adaptedThumbsUp: 0,
      adaptedThumbsDown: 0,
      baseThumbsUp: 0,
      baseThumbsDown: 0,
      editsProvided: 0,
    }
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  subtext,
  highlight,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  subtext?: string;
  highlight?: boolean;
}) {
  return (
    <Card className={cn("p-4", highlight && "border-primary bg-primary/5")}>
      <div className="flex items-start gap-3">
        <div className={cn(
          "p-2 rounded-lg",
          highlight ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
        )}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-2xl font-bold">{value}</p>
          {subtext && <p className="text-xs text-muted-foreground mt-1">{subtext}</p>}
        </div>
      </div>
    </Card>
  );
}

function WinRateBar({ adapted, base, ties }: { adapted: number; base: number; ties: number }) {
  const total = adapted + base + ties;
  if (total === 0) return null;

  const adaptedPct = (adapted / total) * 100;
  const tiesPct = (ties / total) * 100;
  const basePct = (base / total) * 100;

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>Your Model</span>
        <span>Base Model</span>
      </div>
      <div className="h-4 rounded-full overflow-hidden flex bg-muted">
        <div
          className="bg-green-500 transition-all"
          style={{ width: `${adaptedPct}%` }}
        />
        <div
          className="bg-yellow-400 transition-all"
          style={{ width: `${tiesPct}%` }}
        />
        <div
          className="bg-slate-400 transition-all"
          style={{ width: `${basePct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{adapted} wins ({adaptedPct.toFixed(0)}%)</span>
        <span>{ties} ties</span>
        <span>{base} wins ({basePct.toFixed(0)}%)</span>
      </div>
    </div>
  );
}

function AdaptationCard({ adaptation }: { adaptation: UserAdaptation }) {
  const levelColors: Record<string, string> = {
    rag: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    soft_prompt: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
    lora: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  };

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Badge className={levelColors[adaptation.level] || ""}>
            {adaptation.level.toUpperCase()}
          </Badge>
          <Badge variant="outline">{adaptation.status}</Badge>
        </div>
        <span className="text-xs text-muted-foreground">
          #{adaptation.id}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-muted-foreground">Contributions</p>
          <p className="font-semibold">{adaptation.contribution_count}</p>
        </div>
        <div>
          <p className="text-muted-foreground">Created</p>
          <p className="font-semibold">
            {new Date(adaptation.created_at).toLocaleDateString()}
          </p>
        </div>
      </div>
      {adaptation.metrics && Object.keys(adaptation.metrics).length > 0 && (
        <div className="mt-3 pt-3 border-t">
          <p className="text-xs text-muted-foreground mb-2">Metrics</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(adaptation.metrics).map(([key, value]) => (
              <Badge key={key} variant="secondary" className="text-xs">
                {key}: {String(value)}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

export default function ImpactPage() {
  const { user } = useAuth();
  const [adaptations, setAdaptations] = useState<UserAdaptation[]>([]);
  const [feedback, setFeedback] = useState<ComparisonFeedback[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!user) return;

    Promise.all([
      api.getAdaptations(),
      api.getComparisonFeedback(),
    ])
      .then(([adaptationsData, feedbackData]) => {
        setAdaptations(adaptationsData);
        setFeedback(feedbackData);
      })
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, [user]);

  const stats = useMemo(() => computeStats(feedback), [feedback]);

  const totalContributions = useMemo(
    () => adaptations.reduce((sum, a) => sum + a.contribution_count, 0),
    [adaptations]
  );

  const readyAdaptations = adaptations.filter((a) => a.status === "ready");
  const winRate = stats.totalComparisons > 0
    ? ((stats.adaptedPreferred / stats.totalComparisons) * 100).toFixed(0)
    : null;

  if (isLoading) {
    return (
      <main className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </main>
    );
  }

  return (
    <main className="p-4">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold">Your Impact</h1>
          <p className="text-muted-foreground">
            See how your contributions improve model performance
          </p>
        </div>

        {/* Key metrics */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            icon={Layers}
            label="Total Contributions"
            value={totalContributions}
            subtext="Images & feedback"
          />
          <StatCard
            icon={Sparkles}
            label="Active Adaptations"
            value={readyAdaptations.length}
            subtext={`${adaptations.length} total`}
          />
          <StatCard
            icon={Target}
            label="Comparisons Made"
            value={stats.totalComparisons}
            subtext="Side-by-side evaluations"
          />
          <StatCard
            icon={Award}
            label="Your Model Wins"
            value={winRate ? `${winRate}%` : "—"}
            subtext={stats.adaptedPreferred > 0 ? `${stats.adaptedPreferred} preferences` : "Run comparisons to see"}
            highlight={Number(winRate) > 50}
          />
        </div>

        {/* Win rate visualization */}
        {stats.totalComparisons > 0 && (
          <Card className="p-6">
            <h2 className="font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Preference Distribution
            </h2>
            <WinRateBar
              adapted={stats.adaptedPreferred}
              base={stats.basePreferred}
              ties={stats.ties}
            />
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Your Model Thumbs</p>
                <p>
                  <span className="text-green-600">👍 {stats.adaptedThumbsUp}</span>
                  {" / "}
                  <span className="text-red-600">👎 {stats.adaptedThumbsDown}</span>
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Base Model Thumbs</p>
                <p>
                  <span className="text-green-600">👍 {stats.baseThumbsUp}</span>
                  {" / "}
                  <span className="text-red-600">👎 {stats.baseThumbsDown}</span>
                </p>
              </div>
            </div>
          </Card>
        )}

        {/* Adaptations list */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Your Adaptations
            </h2>
            <Link href="/compare">
              <Button variant="outline" size="sm">
                Run Comparison
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
          </div>

          {adaptations.length === 0 ? (
            <Card className="p-8 text-center">
              <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="font-semibold mb-2">No adaptations yet</h3>
              <p className="text-muted-foreground mb-4">
                Contribute images and feedback to create personalized model variants
              </p>
              <Link href="/chat">
                <Button>Start Contributing</Button>
              </Link>
            </Card>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {adaptations.map((adaptation) => (
                <AdaptationCard key={adaptation.id} adaptation={adaptation} />
              ))}
            </div>
          )}
        </div>

        {/* CTA for more comparisons */}
        {stats.totalComparisons > 0 && stats.totalComparisons < 10 && (
          <Card className="p-6 bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold">Keep comparing!</h3>
                <p className="text-sm text-muted-foreground">
                  More comparisons help us understand your preferences better
                </p>
              </div>
              <Link href="/compare">
                <Button>
                  Compare Now
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Button>
              </Link>
            </div>
          </Card>
        )}
      </div>
    </main>
  );
}