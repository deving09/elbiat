"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { api, getAccessToken, UserAdaptation, ComparisonResponse } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import {
  Send,
  Image as ImageIcon,
  X,
  Loader2,
  Globe,
  ThumbsUp,
  ThumbsDown,
  Edit3,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Minus,
  RefreshCw,
} from "lucide-react";

const RANDOM_PUBLIC_IMAGE = "/images/random_public";

type Preference = "base" | "adapted" | "tie" | null;
type Thumbs = "up" | "down" | null;

export default function ComparePage() {
  const { user } = useAuth();

  // Image state
  const [imageId, setImageId] = useState<number | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isPickingImage, setIsPickingImage] = useState(false);

  // Prompt & responses
  const [prompt, setPrompt] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);

  // Adaptation selection
  const [adaptations, setAdaptations] = useState<UserAdaptation[]>([]);
  const [selectedAdaptationId, setSelectedAdaptationId] = useState<number | null>(null);

  // Feedback state
  const [preference, setPreference] = useState<Preference>(null);
  const [baseThumbs, setBaseThumbs] = useState<Thumbs>(null);
  const [adaptedThumbs, setAdaptedThumbs] = useState<Thumbs>(null);
  const [editTarget, setEditTarget] = useState<"base" | "adapted" | null>(null);
  const [editText, setEditText] = useState("");
  const [comment, setComment] = useState("");
  const [showComment, setShowComment] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  // Load user's ready adaptations
  useEffect(() => {
    if (!user) return;
    api.getAdaptations(undefined, "ready").then(setAdaptations).catch(console.error);
  }, [user]);

  const getAuthHeaders = (): Record<string, string> => {
    const token = getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const pickRandomImage = async () => {
    try {
      setIsPickingImage(true);
      const resp = await fetch(RANDOM_PUBLIC_IMAGE, { headers: getAuthHeaders() });
      if (!resp.ok) throw new Error("Failed to fetch random image");
      const img = await resp.json();
      setImageId(img.id);
      setImagePreview(img.file_url ?? `/images/${img.id}/file`);
      resetComparison();
    } catch (e) {
      console.error(e);
    } finally {
      setIsPickingImage(false);
    }
  };

  const resetComparison = () => {
    setComparison(null);
    setPreference(null);
    setBaseThumbs(null);
    setAdaptedThumbs(null);
    setEditTarget(null);
    setEditText("");
    setComment("");
    setShowComment(false);
    setFeedbackSubmitted(false);
  };

  const runComparison = async () => {
    if (!imageId || !prompt.trim()) return;

    setIsRunning(true);
    resetComparison();

    try {
      const result = await api.runComparison({
        image_id: imageId,
        prompt: prompt.trim(),
        adaptation_id: selectedAdaptationId ?? undefined,
      });
      setComparison(result);
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunning(false);
    }
  };

  const submitFeedback = async () => {
    if (!comparison || !imageId) return;

    setIsSubmitting(true);
    try {
      await api.submitComparisonFeedback({
        adaptation_id: comparison.adaptation_id,
        image_id: imageId,
        prompt,
        base_response: comparison.base_response,
        adapted_response: comparison.adapted_response,
        preference: preference ?? undefined,
        base_thumbs: baseThumbs ?? undefined,
        adapted_thumbs: adaptedThumbs ?? undefined,
        edit_target: editTarget ?? undefined,
        edit_original: editTarget ? (editTarget === "base" ? comparison.base_response : comparison.adapted_response) : undefined,
        edit_revised: editTarget ? editText : undefined,
        comment: comment || undefined,
      });
      setFeedbackSubmitted(true);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleEdit = (target: "base" | "adapted") => {
    if (!comparison) return;
    setEditTarget(target);
    setEditText(target === "base" ? comparison.base_response : comparison.adapted_response);
  };

  const hasFeedback = preference || baseThumbs || adaptedThumbs || editTarget || comment;

  return (
    <main className="flex flex-col min-h-[calc(100vh-4rem)] p-4">
      <div className="mx-auto max-w-5xl w-full space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold">Compare Models</h1>
          <p className="text-muted-foreground">
            See how your personalized model compares to the base model
          </p>
        </div>

        {/* Setup row */}
        <Card className="p-4">
          <div className="flex flex-wrap gap-4 items-end">
            {/* Image selection */}
            <div className="flex-1 min-w-[200px]">
              <label className="text-sm font-medium mb-2 block">Image</label>
              {imagePreview ? (
                <div className="flex items-center gap-3">
                  <img
                    src={imagePreview}
                    alt="Selected"
                    className="h-16 w-16 object-cover rounded-lg border"
                  />
                  <div className="flex flex-col gap-1">
                    <span className="text-sm text-muted-foreground">Image #{imageId}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={pickRandomImage}
                      disabled={isPickingImage}
                      className="h-7"
                    >
                      <RefreshCw className={cn("h-3 w-3 mr-1", isPickingImage && "animate-spin")} />
                      Different image
                    </Button>
                  </div>
                </div>
              ) : (
                <Button
                  variant="outline"
                  onClick={pickRandomImage}
                  disabled={isPickingImage}
                  className="w-full justify-start"
                >
                  {isPickingImage ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Globe className="h-4 w-4 mr-2" />
                  )}
                  Pick random public image
                </Button>
              )}
            </div>

            {/* Adaptation selection */}
            <div className="flex-1 min-w-[200px]">
              <label className="text-sm font-medium mb-2 block">Your Adaptation</label>
              <select
                className="w-full h-10 px-3 rounded-md border bg-background text-sm"
                value={selectedAdaptationId ?? ""}
                onChange={(e) => setSelectedAdaptationId(e.target.value ? Number(e.target.value) : null)}
              >
                <option value="">Latest ready adaptation</option>
                {adaptations.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.level.toUpperCase()} (#{a.id})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Prompt input */}
          <div className="mt-4">
            <label className="text-sm font-medium mb-2 block">Prompt</label>
            <div className="flex gap-2">
              <Input
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="What would you like to ask about this image?"
                className="flex-1"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    runComparison();
                  }
                }}
              />
              <Button
                onClick={runComparison}
                disabled={!imageId || !prompt.trim() || isRunning}
              >
                {isRunning ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <>
                    <Send className="h-4 w-4 mr-2" />
                    Compare
                  </>
                )}
              </Button>
            </div>
          </div>
        </Card>

        {/* Results - side by side */}
        {(comparison || isRunning) && (
          <div className="grid md:grid-cols-2 gap-4">
            {/* Base model response */}
            <Card className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">Base Model</h3>
                {comparison && (
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className={cn("h-8 w-8 p-0", baseThumbs === "up" && "text-green-500")}
                      onClick={() => setBaseThumbs(baseThumbs === "up" ? null : "up")}
                    >
                      <ThumbsUp className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={cn("h-8 w-8 p-0", baseThumbs === "down" && "text-red-500")}
                      onClick={() => setBaseThumbs(baseThumbs === "down" ? null : "down")}
                    >
                      <ThumbsDown className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={cn("h-8 w-8 p-0", editTarget === "base" && "text-blue-500")}
                      onClick={() => handleEdit("base")}
                    >
                      <Edit3 className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
              {isRunning ? (
                <div className="flex items-center justify-center h-32">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <p className="text-sm whitespace-pre-wrap">{comparison?.base_response}</p>
              )}
            </Card>

            {/* Adapted model response */}
            <Card className="p-4 border-primary/50">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold">Your Model</h3>
                  {comparison && (
                    <Badge variant="secondary" className="text-xs">
                      {comparison.adaptation_level}
                    </Badge>
                  )}
                </div>
                {comparison && (
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className={cn("h-8 w-8 p-0", adaptedThumbs === "up" && "text-green-500")}
                      onClick={() => setAdaptedThumbs(adaptedThumbs === "up" ? null : "up")}
                    >
                      <ThumbsUp className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={cn("h-8 w-8 p-0", adaptedThumbs === "down" && "text-red-500")}
                      onClick={() => setAdaptedThumbs(adaptedThumbs === "down" ? null : "down")}
                    >
                      <ThumbsDown className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={cn("h-8 w-8 p-0", editTarget === "adapted" && "text-blue-500")}
                      onClick={() => handleEdit("adapted")}
                    >
                      <Edit3 className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
              {isRunning ? (
                <div className="flex items-center justify-center h-32">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <p className="text-sm whitespace-pre-wrap">{comparison?.adapted_response}</p>
              )}
            </Card>
          </div>
        )}

        {/* Preference selector */}
        {comparison && !feedbackSubmitted && (
          <Card className="p-4">
            <h3 className="font-semibold mb-3">Which response was better?</h3>
            <div className="flex items-center justify-center gap-2">
              <Button
                variant={preference === "base" ? "default" : "outline"}
                onClick={() => setPreference(preference === "base" ? null : "base")}
                className="flex-1 max-w-[150px]"
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Base
              </Button>
              <Button
                variant={preference === "tie" ? "default" : "outline"}
                onClick={() => setPreference(preference === "tie" ? null : "tie")}
                className="flex-1 max-w-[150px]"
              >
                <Minus className="h-4 w-4 mr-1" />
                Tie
              </Button>
              <Button
                variant={preference === "adapted" ? "default" : "outline"}
                onClick={() => setPreference(preference === "adapted" ? null : "adapted")}
                className="flex-1 max-w-[150px]"
              >
                Yours
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>

            {/* Edit area */}
            {editTarget && (
              <div className="mt-4">
                <label className="text-sm font-medium mb-2 block">
                  Edit the {editTarget === "base" ? "base" : "adapted"} response:
                </label>
                <Textarea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  rows={4}
                  className="text-sm"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-1"
                  onClick={() => {
                    setEditTarget(null);
                    setEditText("");
                  }}
                >
                  Cancel edit
                </Button>
              </div>
            )}

            {/* Comment toggle */}
            <div className="mt-4">
              {showComment ? (
                <div>
                  <label className="text-sm font-medium mb-2 block">Comment (optional)</label>
                  <Textarea
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="Any additional thoughts..."
                    rows={2}
                    className="text-sm"
                  />
                </div>
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowComment(true)}
                >
                  <MessageSquare className="h-4 w-4 mr-1" />
                  Add comment
                </Button>
              )}
            </div>

            {/* Submit */}
            <div className="mt-4 flex justify-end">
              <Button
                onClick={submitFeedback}
                disabled={!hasFeedback || isSubmitting}
              >
                {isSubmitting ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : null}
                Submit Feedback
              </Button>
            </div>
          </Card>
        )}

        {/* Success message */}
        {feedbackSubmitted && (
          <Card className="p-4 bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800">
            <p className="text-green-800 dark:text-green-200 text-center">
              Feedback submitted! This helps improve your personalized model.
            </p>
            <div className="flex justify-center mt-3">
              <Button variant="outline" onClick={resetComparison}>
                Compare again
              </Button>
            </div>
          </Card>
        )}

        {/* Empty state */}
        {!comparison && !isRunning && (
          <div className="text-center py-12 text-muted-foreground">
            <p>Select an image and enter a prompt to compare model responses</p>
          </div>
        )}
      </div>
    </main>
  );
}