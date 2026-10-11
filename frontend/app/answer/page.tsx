"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { getAccessToken } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import {
  Send,
  X,
  Loader2,
  RefreshCw,
  MessageCircleQuestion,
  Construction,
} from "lucide-react";

const RANDOM_PUBLIC_IMAGE = "/images/random_public";



export default function AnswerPage() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageId, setImageId] = useState<number | null>(null);
  const [isLoadingImage, setIsLoadingImage] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [expandedSrc, setExpandedSrc] = useState<string | null>(null);

  const { user } = useAuth();

  const getAuthHeaders = (): Record<string, string> => {
    const token = getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const loadNewQuestion = async () => {
    try {
      setIsLoadingImage(true);
      setSubmitted(false);
      setAnswer("");

      const resp = await fetch("/api/questions/next", {
        headers: getAuthHeaders(),
      });

      if (!resp.ok) {
        throw new Error("Failed to fetch question");
      }

      const data = await resp.json();
      setImageId(data.image_id);
      setImagePreview(data.image_path ? `/images/${data.image_id}/file` : null);
      setQuestion(data.question_text);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoadingImage(false);
    }
  };

  useEffect(() => {
    loadNewQuestion();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!answer.trim() || !imageId) return;

    setIsSubmitting(true);

    try {
      const resp = await fetch(`/api/questions/${imageId}/answer`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify({ answer_text: answer }),
      });

      if (!resp.ok) {
        throw new Error("Failed to submit answer");
      }

      setSubmitted(true);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="border-b bg-muted/30 p-4">
        <div className="mx-auto max-w-3xl">
          <h1 className="text-lg font-semibold flex items-center gap-2">
            <MessageCircleQuestion className="h-5 w-5" />
            Answer a Question
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Help train the model by answering questions it's uncertain about.
          </p>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl">
          <Card className="p-6">
            {/* Image */}
            <div className="flex justify-center mb-6">
              {isLoadingImage ? (
                <div className="h-64 w-full max-w-md rounded-lg bg-muted flex items-center justify-center">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Question image"
                  className="max-h-64 rounded-lg border cursor-zoom-in"
                  onClick={() => setExpandedSrc(imagePreview)}
                />
              ) : (
                <div className="h-64 w-full max-w-md rounded-lg bg-muted flex items-center justify-center">
                  <MessageCircleQuestion className="h-12 w-12 text-muted-foreground" />
                </div>
              )}
            </div>

            {/* Question */}
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-6">
              <p className="text-sm font-medium text-primary mb-1">Model's Question:</p>
              <p className="text-lg">{question || "Loading question..."}</p>
            </div>

            {/* Answer form */}
            {!submitted ? (
              <form onSubmit={handleSubmit}>
                <Textarea
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Type your answer here..."
                  className="min-h-[100px] mb-4"
                  disabled={isSubmitting || !imageId}
                />
                <div className="flex justify-between">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={loadNewQuestion}
                    disabled={isLoadingImage || isSubmitting}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Skip / New Question
                  </Button>
                  <Button type="submit" disabled={isSubmitting || !answer.trim()}>
                    {isSubmitting ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4 mr-2" />
                    )}
                    Submit Answer
                  </Button>
                </div>
              </form>
            ) : (
              <div className="text-center py-6">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-100 dark:bg-green-900 mb-3">
                  <span className="text-2xl">✓</span>
                </div>
                <p className="text-lg font-medium mb-2">Thanks for your answer!</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Your response will help improve the model's understanding.
                </p>
                <Button onClick={loadNewQuestion}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Next Question
                </Button>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Image lightbox */}
      {expandedSrc && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-start justify-center p-4 overflow-y-auto"
          onClick={() => setExpandedSrc(null)}
        >
          <div className="relative max-w-6xl w-full my-8" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setExpandedSrc(null)}
              className="absolute top-2 right-2 bg-white/10 hover:bg-white/20 text-white rounded-full p-2 z-10"
            >
              <X className="h-5 w-5" />
            </button>
            <img 
              src={expandedSrc} 
              alt="Expanded" 
              className="w-full h-auto max-h-none rounded-lg" 
            />
          </div>
        </div>
      )}
    </main>
  );
}