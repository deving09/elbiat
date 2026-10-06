"use client";

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ArrowLeft,
  Check,
  X,
  Loader2,
  Star,
  Calendar,
  Cpu,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getAccessToken } from "@/lib/api";
import { useState } from "react";

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

const fetchFeedback = async (id: string): Promise<Convo> => {
  const token = getAccessToken();
  const response = await fetch(`/api/feedback/${id}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error("Failed to fetch feedback");
  return response.json();
};

const updateFeedback = async (
  id: number,
  data: { feedback?: string; enabled?: boolean }
): Promise<Convo> => {
  const token = getAccessToken();
  const response = await fetch(`/api/feedback/${id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error("Failed to update feedback");
  return response.json();
};

const getScoreColor = (score: number | null): string => {
  if (score === null) return "bg-gray-500";
  if (score > 0.03) return "bg-green-500";
  if (score > 0) return "bg-green-400";
  if (score > -0.01) return "bg-yellow-500";
  return "bg-red-500";
};

export default function FeedbackDetailPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const id = params.id as string;

  const [isEditing, setIsEditing] = useState(false);
  const [editedFeedback, setEditedFeedback] = useState("");

  const { data: convo, isLoading, error } = useQuery({
    queryKey: ["feedback", id],
    queryFn: () => fetchFeedback(id),
    enabled: !!id,
  });

  const updateMutation = useMutation({
    mutationFn: (data: { feedback?: string; enabled?: boolean }) =>
      updateFeedback(parseInt(id), data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feedback"] });
    },
  });

  const handleBack = () => {
    // Preserve filters when going back
    const backUrl = searchParams.get("from") || "/feedback";
    router.push(backUrl);
  };

  const startEditing = () => {
    if (convo) {
      setEditedFeedback(convo.feedback);
      setIsEditing(true);
    }
  };

  const cancelEditing = () => {
    setIsEditing(false);
    setEditedFeedback("");
  };

  const saveFeedback = () => {
    if (convo && editedFeedback !== convo.feedback) {
      updateMutation.mutate({ feedback: editedFeedback });
    }
    setIsEditing(false);
  };

  const toggleEnabled = () => {
    if (convo) {
      updateMutation.mutate({ enabled: !convo.enabled });
    }
  };

  if (isLoading) {
    return (
      <main className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </main>
    );
  }

  if (error || !convo) {
    return (
      <main className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 py-8">
        <Button variant="ghost" onClick={handleBack} className="mb-4">
          <ArrowLeft className="h-4 w-4 mr-2" /> Back
        </Button>
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-lg text-muted-foreground">Feedback not found</p>
          </CardContent>
        </Card>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={handleBack}>
            <ArrowLeft className="h-4 w-4 mr-2" /> Back
          </Button>
          <h1 className="text-2xl font-bold">Feedback #{convo.id}</h1>
          <Badge variant={convo.enabled ? "default" : "secondary"}>
            {convo.enabled ? "Enabled" : "Disabled"}
          </Badge>
        </div>
        <Button
          variant="outline"
          onClick={toggleEnabled}
          disabled={updateMutation.isPending}
        >
          {convo.enabled ? (
            <><X className="h-4 w-4 mr-2" /> Disable</>
          ) : (
            <><Check className="h-4 w-4 mr-2" /> Enable</>
          )}
        </Button>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left Column - Image */}
        <div className="space-y-6">
          <Card>
            <CardContent className="p-4">
              <img
                src={`/images/${convo.image_id}/file`}
                alt={`Image ${convo.image_id}`}
                className="w-full rounded-lg"
              />
            </CardContent>
          </Card>

          {/* Metadata */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Cpu className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Model:</span>
                <span>{convo.model_name}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <FileText className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Task:</span>
                <span>{convo.task}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Calendar className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Created:</span>
                <span>{new Date(convo.created_at).toLocaleString()}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Content */}
        <div className="space-y-6">
          {/* Prompt */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Prompt</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm whitespace-pre-wrap">{convo.prompt}</p>
            </CardContent>
          </Card>

          {/* Model Response */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Model Response</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm whitespace-pre-wrap">{convo.response}</p>
            </CardContent>
          </Card>

          {/* Feedback */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg">Your Feedback</CardTitle>
              {!isEditing && (
                <Button variant="outline" size="sm" onClick={startEditing}>
                  Edit
                </Button>
              )}
            </CardHeader>
            <CardContent>
              {isEditing ? (
                <div className="space-y-3">
                  <Textarea
                    value={editedFeedback}
                    onChange={(e) => setEditedFeedback(e.target.value)}
                    rows={6}
                  />
                  <div className="flex gap-2 justify-end">
                    <Button variant="outline" onClick={cancelEditing}>
                      Cancel
                    </Button>
                    <Button onClick={saveFeedback} disabled={updateMutation.isPending}>
                      {updateMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Check className="h-4 w-4 mr-2" />
                      )}
                      Save
                    </Button>
                  </div>
                </div>
              ) : (
                <p className="text-sm whitespace-pre-wrap">{convo.feedback}</p>
              )}
            </CardContent>
          </Card>

          {/* Attribution Scores */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Star className="h-5 w-5" />
                Attribution Scores
              </CardTitle>
            </CardHeader>
            <CardContent>
              {convo.attributions.length > 0 ? (
                <div className="space-y-3">
                  {convo.attributions.map((attr, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between p-3 bg-muted rounded-lg"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{attr.benchmark}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {attr.method}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Checkpoint: {attr.checkpoint_id}
                        </p>
                      </div>
                      <div className="text-right">
                        <Badge
                          className={cn(
                            "text-white text-sm",
                            getScoreColor(attr.influence_score)
                          )}
                        >
                          {attr.influence_score.toFixed(6)}
                        </Badge>
                        {attr.rank && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Rank #{attr.rank}
                          </p>
                        )}
                        {attr.z_score && (
                          <p className="text-xs text-muted-foreground">
                            z-score: {attr.z_score.toFixed(2)}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No attribution scores computed yet
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}