"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { getAccessToken } from "@/lib/api";
import { FeedbackControls } from "@/components/chat/FeedbackControls";
import { useAuth } from "@/lib/auth";
import {
  Send,
  X,
  User,
  Bot,
  Loader2,
  RefreshCw,
  HelpCircle,
  Paperclip,
  Link,
  Image as ImageIcon,
  Globe,
} from "lucide-react";

const RANDOM_PUBLIC_IMAGE = "/images/random_public";
const CHAT_ENDPOINT = "/api/chat";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export default function AskPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageId, setImageId] = useState<number | null>(null);
  const [isLoadingImage, setIsLoadingImage] = useState(false);
  const [historyState, setHistoryState] = useState<any>(null);
  const [convoId, setConvoId] = useState<number | null>(null);
  const [expandedSrc, setExpandedSrc] = useState<string | null>(null);

  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const attachMenuRef = useRef<HTMLDivElement>(null);

  const { user } = useAuth();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getAuthHeaders = (): Record<string, string> => {
    const token = getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  // Load a random image on mount
  useEffect(() => {
    loadRandomImage();
  }, []);

  const loadRandomImage = async () => {
    try {
      setIsLoadingImage(true);
      const resp = await fetch(RANDOM_PUBLIC_IMAGE, {
        headers: getAuthHeaders(),
      });

      if (!resp.ok) {
        throw new Error("Failed to fetch random image");
      }

      const img = await resp.json();
      const id = Number(img.id);
      if (!Number.isFinite(id)) throw new Error("Bad image payload");

      setImageId(id);
      setImagePreview(img.file_url ?? `/images/${id}/file`);
      setHistoryState(null);
      setMessages([]);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoadingImage(false);
    }
  };

  // Close attach menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (attachMenuRef.current && !attachMenuRef.current.contains(e.target as Node)) {
        setShowAttachMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleUrlSubmit = async () => {
    if (!imageUrl.trim()) return;
    setImagePreview(imageUrl);
    setSelectedFile(null);
    setImageId(null);
    setHistoryState(null);
    setMessages([]);
    setShowUrlInput(false);
    setShowAttachMenu(false);
  };

  const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setImageUrl("");
    setImageId(null);
    setHistoryState(null);
    setMessages([]);
    setShowAttachMenu(false);

    const reader = new FileReader();
    reader.onload = (ev) => {
      setImagePreview(ev.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Auto-upload
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("is_public", "false");

      const headers = getAuthHeaders();
      delete (headers as any)["Content-Type"];

      const response = await fetch("/images/ingest_upload", {
        method: "POST",
        headers,
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setImageId(data.image_id);
      }
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const removeImage = () => {
    setSelectedFile(null);
    setImageUrl("");
    setImagePreview(null);
    setImageId(null);
    setHistoryState(null);
    setMessages([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !imageId) return;

    const token = getAccessToken();
    if (!token) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "❌ Please log in first.",
        },
      ]);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const chatPayload = {
        prompt: input,
        image_id: imageId,
        history: historyState,
        max_new_tokens: 1024,
        do_sample: false,
        return_history: true,
      };

      const response = await fetch(CHAT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify(chatPayload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.detail?.upstream || errorData.detail || "Chat failed";
        throw new Error(typeof errorMsg === "string" ? errorMsg : JSON.stringify(errorMsg));
      }

      const data = await response.json();
      setHistoryState(data.history || null);

      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.response || "",
        },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `❌ ${error instanceof Error ? error.message : "An error occurred"}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header with image */}
      <div className="border-b bg-muted/30 p-4">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-center gap-4">
            {isLoadingImage ? (
              <div className="h-32 w-32 rounded-lg bg-muted flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : imagePreview ? (
              <img
                src={imagePreview}
                alt="Question target"
                className="h-32 w-32 object-cover rounded-lg border cursor-zoom-in"
                onClick={() => setExpandedSrc(imagePreview)}
              />
            ) : (
              <div className="h-32 w-32 rounded-lg bg-muted flex items-center justify-center">
                <HelpCircle className="h-8 w-8 text-muted-foreground" />
              </div>
            )}
            <div className="flex-1">
              <h1 className="text-lg font-semibold flex items-center gap-2">
                <HelpCircle className="h-5 w-5" />
                Ask a Question
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Ask a question that would require <strong>reasoning</strong> to answer.
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Think about: spatial relationships, counting, comparisons, or inferences that aren't immediately obvious.
              </p>
              <Button
                variant="outline"
                size="sm"
                className="mt-3"
                onClick={loadRandomImage}
                disabled={isLoadingImage}
              >
                {isLoadingImage ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4 mr-2" />
                )}
                New Image
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <HelpCircle className="h-16 w-16 text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">What do you want to know?</h2>
              <p className="text-muted-foreground max-w-sm">
                Look at the image above and ask a question that requires careful observation or reasoning to answer.
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex items-start space-x-3",
                  message.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.role === "assistant" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <Bot className="h-5 w-5 text-primary-foreground" />
                  </div>
                )}
                <div className="flex flex-col">
                  <Card
                    className={cn(
                      "max-w-[80%] p-4",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-card"
                    )}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </Card>
                  {message.role === "assistant" && user && (
                    <FeedbackControls
                      convoId={convoId ?? undefined}
                      turnIndex={messages.findIndex((m) => m.id === message.id)}
                      messageContent={message.content}
                      messages={messages.map((m) => ({ role: m.role, content: m.content }))}
                      onFeedbackSubmit={(newConvoId) => setConvoId(newConvoId)}
                    />
                  )}
                </div>
                {message.role === "user" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                    <User className="h-5 w-5 text-secondary-foreground" />
                  </div>
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                <Bot className="h-5 w-5 text-primary-foreground" />
              </div>
              <Card className="p-4">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </Card>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="border-t bg-background p-4">
        <div className="mx-auto max-w-3xl">
          {/* URL input popover */}
          {showUrlInput && (
            <div className="mb-3 flex gap-2 items-center p-3 bg-muted rounded-lg">
              <Link className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              <Input
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="Paste image URL..."
                className="flex-1 h-8"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    handleUrlSubmit();
                  } else if (e.key === "Escape") {
                    setShowUrlInput(false);
                    setImageUrl("");
                  }
                }}
              />
              <Button size="sm" variant="ghost" onClick={handleUrlSubmit} className="h-8">
                Add
              </Button>
              <Button size="sm" variant="ghost" onClick={() => { setShowUrlInput(false); setImageUrl(""); }} className="h-8">
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            {/* Paperclip attachment button */}
            <div className="relative" ref={attachMenuRef}>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-10 w-10 rounded-full"
                onClick={() => setShowAttachMenu(!showAttachMenu)}
              >
                <Paperclip className="h-5 w-5 text-muted-foreground" />
              </Button>

              {showAttachMenu && (
                <div className="absolute bottom-12 left-0 bg-popover border rounded-lg shadow-lg py-1 min-w-[160px] z-10">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageSelect}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-sm text-left hover:bg-muted flex items-center gap-2"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <ImageIcon className="h-4 w-4" />
                    Upload image
                  </button>
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-sm text-left hover:bg-muted flex items-center gap-2"
                    onClick={() => {
                      setShowAttachMenu(false);
                      setShowUrlInput(true);
                    }}
                  >
                    <Link className="h-4 w-4" />
                    Image URL
                  </button>
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-sm text-left hover:bg-muted flex items-center gap-2"
                    onClick={() => {
                      setShowAttachMenu(false);
                      loadRandomImage();
                    }}
                    disabled={isLoadingImage}
                  >
                    {isLoadingImage ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Globe className="h-4 w-4" />
                    )}
                    Random public image
                  </button>
                </div>
              )}
            </div>

            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a reasoning question about the image..."
              className="flex-1"
              disabled={isLoading || !imageId}
            />

            <Button
              type="submit"
              size="icon"
              disabled={isLoading || !input.trim() || !imageId}
              className="h-10 w-10 rounded-full"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </Button>
          </form>
        </div>
      </div>

      {/* Image lightbox */}
      {expandedSrc && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setExpandedSrc(null)}
        >
          <div className="relative max-w-4xl w-full" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setExpandedSrc(null)}
              className="absolute top-2 right-2 bg-white/10 hover:bg-white/20 text-white rounded-full p-2"
            >
              <X className="h-5 w-5" />
            </button>
            <img src={expandedSrc} alt="Expanded" className="w-full h-auto rounded-lg" />
          </div>
        </div>
      )}
    </main>
  );
}