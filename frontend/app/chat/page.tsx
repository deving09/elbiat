"use client";

import { useState, useRef, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";

//import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { getAccessToken } from "@/lib/api";
import {
  Send,
  Image as ImageIcon,
  X,
  User,
  Bot,
  Loader2,
  Globe,
  ThumbsUp,
  ThumbsDown,
  Save,
} from "lucide-react";

// All calls go through Next.js proxy -> FastAPI
// API Configuration - match your Gradio setup
/*
const DATA_BASE = "http://127.0.0.1:8000";
const MODEL_BASE = "http://127.0.0.1:9000";

const INGEST_URL = `${DATA_BASE}/images/ingest_url`;
const INGEST_UPLOAD = `${DATA_BASE}/images/ingest_upload`;
const META_IMAGE = (id: number) => `${DATA_BASE}/images/${id}/meta`;
const CHAT_ENDPOINT = `${DATA_BASE}/chat/internvl2_5_2b`;
const CONVOS_ENDPOINT = `${DATA_BASE}/convos`;
*/

const INGEST_URL = "/images/ingest_url";
const INGEST_UPLOAD = "/images/ingest_upload";
const META_IMAGE = (id: number) => `/images/${id}/meta`;
const CHAT_ENDPOINT = "/api/chat";
const CONVOS_ENDPOINT = "/convos";
const RANDOM_PUBLIC_IMAGE = "/images/random_public";





interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  imageUrl?: string;
}

export default function ChatPage() {
  const searchParams = useSearchParams();
  const imageIdParam = searchParams.get("imageId");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const [isPickingRandom, setIsPickingRandom] = useState(false);


  // Expanding image sizei
  //const [expandedSrc, setExpandedSrc] = useState<string | null>(null);

  //const [isImageOpen, setIsImageOpen] = useState(false);
  const [expandedSrc, setExpandedSrc] = useState<string | null>(null);
  const [expandedTitle, setExpandedTitle] = useState<string>("Image");

  const openImage = (src: string, title = "Image") => {
    setExpandedSrc(src);
    //setExpandedTitle(title);
    //#setIsImageOpen(true);
  };

  const closeImage = () => {
    setExpandedSrc(null);
  };
  
  // Backend state
  const [imageId, setImageId] = useState<number | null>(null);
  const [historyState, setHistoryState] = useState<any>(null);
  const [lastResponse, setLastResponse] = useState<string>("");
  const [lastPrompt, setLastPrompt] = useState<string>("");
  const [imageMeta, setImageMeta] = useState<any>(null);

  
  // Feedback
  const [feedback, setFeedback] = useState("");
  const [thumbs, setThumbs] = useState<"up" | "down" | null>(null);
  const [saveStatus, setSaveStatus] = useState("");
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  /*useEffect(() => {
    if (!imageIdParam) return;
    const id = Number(imageIdParam);
    if (!Number.isFinite(id)) return;

    setImageId(id);

    // Optional: fetch meta so you can show filename / visibility / etc.
    
  }, [imageIdParam]);
  */

  const pickRandomPublicImage = async () => {
    try {
      setIsPickingRandom(true);
  
      const resp = await fetch(RANDOM_PUBLIC_IMAGE, {
        headers: {
          ...getAuthHeaders(), // ok even if endpoint is public
        },
      });
  
      if (!resp.ok) {
        const err = await resp.text();
        throw new Error(err || "Failed to fetch random public image");
      }
  
      const img = await resp.json();
  
      // Expecting { id, file_url } or at least { id }
      const id = Number(img.id);
      if (!Number.isFinite(id)) throw new Error("Bad image payload from server");
  
      setImageId(id);
      setImagePreview(img.file_url ?? `/images/${id}/file`);
      setSelectedFile(null);
      setImageUrl("");
      setHistoryState(null);
  
      // Optional: start fresh chat
      setMessages([]);
      setLastResponse("");
      setLastPrompt("");
      setFeedback("");
      setThumbs(null);
      setSaveStatus("");
    } catch (e) {
      console.error(e);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: `❌ ${e instanceof Error ? e.message : "Failed to pick random image"}`,
        },
      ]);
    } finally {
      setIsPickingRandom(false);
    }
  };



  useEffect(() => {
    scrollToBottom();
    if (!imageIdParam) return;
    const id = Number(imageIdParam);
    if (!Number.isFinite(id)) return;

    setImageId(id);
    setImagePreview(`/images/${id}/file`);   // so the preview renders
    setSelectedFile(null);
    setImageUrl("");
    // keep historyState if you want continuity; or reset:
    setHistoryState(null);

  }, [imageIdParam]);

  const getAuthHeaders = (): Record<string, string> => {
    const token = getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setImageUrl("");
      setImageId(null);
      setHistoryState(null);
    }
  };

  const handleUrlChange = (url: string) => {
    setImageUrl(url);
    if (url) {
      setImagePreview(url);
      setSelectedFile(null);
      setImageId(null);
      setHistoryState(null);
    }
  };

  const removeImage = () => {
    setSelectedFile(null);
    setImageUrl("");
    setImagePreview(null);
    setImageId(null);
    setHistoryState(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const ingestImage = async (): Promise<number | null> => {
    const token = getAccessToken();
    if (!token) {
      throw new Error("Please log in first");
    }

    if (imageUrl) {
      // Ingest via URL
      const response = await fetch(INGEST_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify({ image_url: imageUrl }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Ingest failed: ${error}`);
      }

      const data = await response.json();
      /*
      if (data.image_path) {
        setImagePreview(data.image_path);
      } 
      */
      if (data.image_id) {
        setImagePreview(`/images/${data.image_id}/file`)
      }
      return data.image_id;

    } else if (selectedFile) {
      // Ingest via upload
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(INGEST_UPLOAD, {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData,
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Upload failed: ${error}`);
      }

      const data = await response.json();
      /*
      if (data.image_path) {
        setImagePreview(data.image_path);
      }
      */
      if (data.image_id) {
        setImagePreview(`/images/${data.image_id}/file`)
      }

      return data.image_id;
    }

    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

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
      imageUrl: !imageId ? imagePreview || undefined : undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLastPrompt(input);
    setInput("");
    setIsLoading(true);
    setFeedback("");
    setThumbs(null);
    setSaveStatus("");

    try {
      // Ingest image if we don't have an image_id yet
      let currentImageId = imageId;
      if (!currentImageId && (imageUrl || selectedFile)) {
        currentImageId = await ingestImage();
        setImageId(currentImageId);
      }

      if (!currentImageId) {
        throw new Error("Please provide an image (URL or upload)");
      }

      // Call FastAPI /api/chat which proxies to model service
      // Payload matches what your model service expects
      const chatPayload = {
        prompt: input,
        image_id: currentImageId,
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
        throw new Error(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
      }

      const data = await response.json();
      const assistantResponse = data.response || "";
      const newHistory = data.history || null;

      setHistoryState(newHistory);
      setLastResponse(assistantResponse);

      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: assistantResponse,
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

  const handleSaveConvo = async () => {
    if (!imageId || !lastPrompt || !lastResponse) {
      setSaveStatus("❌ Need image, prompt, and response before saving");
      return;
    }

    const token = getAccessToken();
    if (!token) {
      setSaveStatus("❌ Please log in first");
      return;
    }

    const conversations = [
      { from: "human", value: `<image>\n${lastPrompt}` },
      { from: "gpt", value: lastResponse },
    ];

    let feedbackText = feedback.trim();
    if (thumbs) {
      feedbackText = `[thumbs=${thumbs}] ${feedbackText}`;
    }

    const payload = {
      image_id: imageId,
      conversations,
      model_name: "internvl2.5_2B",
      model_type: "vlm",
      task: "general_vqa",
      feedback: feedbackText,
      monetized: true,
      enabled: true,
    };

    try {
      const response = await fetch(CONVOS_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error);
      }

      const data = await response.json();
      setSaveStatus(`✅ Saved! convo_id=${data.convo_id || data.id}`);
      setFeedback("");
      setThumbs(null);

    } catch (error) {
      setSaveStatus(`❌ ${error instanceof Error ? error.message : "Save failed"}`);
    }
  };

  return (
    <main className="flex flex-col h-[calc(100vh-4rem)]">
      {imageId && imagePreview && (
        <div className="border-b bg-muted/30 p-3">
          <div className="mx-auto max-w-3xl flex items-center gap-3">
            <img
              src={imagePreview}
              className="h-12 w-12 rounded-md object-cover border cursor-zoom-in"
              onClick={() => openImage(imagePreview, `Image #${imageId}`)}
              alt={`Image ${imageId}`}
            />
            <div className="min-w-0">
              <div className="text-sm font-medium truncate">
                Preloaded image #{imageId}
              </div>
              <div className="text-xs text-muted-foreground">
                From gallery (double-click)
              </div>
            </div>
         </div>
       </div>
     )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <Bot className="h-16 w-16 text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">InternVL2.5-2B Chat</h2>
              <p className="text-muted-foreground max-w-sm">
                Upload an image or provide a URL, then ask questions about it.
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex items-start space-x-3 animate-fade-in",
                  message.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.role === "assistant" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <Bot className="h-5 w-5 text-primary-foreground" />
                  </div>
                )}
                <Card
                  className={cn(
                    "max-w-[80%] p-4",
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-card"
                  )}
                >
                  {message.imageUrl && (
                    <img
                      src={message.imageUrl}
                      alt="Uploaded"
                      className="max-w-full max-h-64 rounded-md mb-2 cursor-zoom-in"
                      onClick={() => openImage(message.imageUrl!, "Message image")}
                    />
                  )}
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </Card>
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

      {/* Feedback section (show after response) */}
      {lastResponse && (
        <div className="border-t bg-muted/30 p-4">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-center gap-4 mb-2">
              <span className="text-sm text-muted-foreground">Rate response:</span>
              <Button
                variant={thumbs === "up" ? "default" : "outline"}
                size="sm"
                onClick={() => setThumbs(thumbs === "up" ? null : "up")}
              >
                <ThumbsUp className="h-4 w-4" />
              </Button>
              <Button
                variant={thumbs === "down" ? "default" : "outline"}
                size="sm"
                onClick={() => setThumbs(thumbs === "down" ? null : "down")}
              >
                <ThumbsDown className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex gap-2">
              <Input
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="What was wrong/right about the answer?"
                className="flex-1"
              />
              <Button onClick={handleSaveConvo}>
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
            </div>
            {saveStatus && (
              <p className={cn(
                "text-sm mt-2",
                saveStatus.startsWith("✅") ? "text-green-600" : "text-destructive"
              )}>
                {saveStatus}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t bg-background p-4">
        <div className="mx-auto max-w-3xl">
          {/* Image input */}
          {!imageId && (
            <div className="mb-3 flex gap-2 items-center">
              <Input
                value={imageUrl}
                onChange={(e) => handleUrlChange(e.target.value)}
                placeholder="Image URL (optional)"
                className="flex-1"
                disabled={!!selectedFile}
              />
              <span className="text-muted-foreground">or</span>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageSelect}
                accept="image/*"
                className="hidden"
              />
              <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={!!imageUrl}
              >
                <ImageIcon className="h-4 w-4 mr-2" />
                Upload
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={pickRandomPublicImage}
                disabled={isLoading || isPickingRandom}
              >
                {isPickingRandom ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Globe className="h-4 w-4 mr-2" />
                )}
                Random Public
            </Button>
            </div>
          )}

          {/* Image preview */}
          {imagePreview && (
            <div className="mb-3 relative inline-block">
              <img
                src={imagePreview}
                alt="Preview"
                className="max-h-32 rounded-md border cursor-zoom-in"
                onClick={() => openImage(imagePreview, imageId ? `Image #${imageId}` : "Preview")}

              />
              {!imageId && (
                <button
                  onClick={removeImage}
                  className="absolute -top-2 -right-2 p-1 rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
              {imageId && (
                <span className="absolute -top-2 -right-2 px-2 py-0.5 rounded-full bg-green-500 text-white text-xs">
                  ID: {imageId}
                </span>
              )}
            </div>
          )}

          {/* Prompt input */}
          <form onSubmit={handleSubmit} className="flex items-center space-x-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about the image..."
              className="flex-1"
              disabled={isLoading}
            />
            <Button type="submit" disabled={isLoading || !input.trim()}>
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </Button>
          </form>

          {/* Reset button */}
          {imageId && (
            <Button
              variant="ghost"
              size="sm"
              className="mt-2"
              onClick={() => {
                removeImage();
                setMessages([]);
                setLastResponse("");
                setLastPrompt("");
              }}
            >
              Start new conversation with different image
            </Button>
          )}
        </div>
      </div>

      {expandedSrc && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={closeImage}
        >
          <div
            className="relative max-w-6xl w-full max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeImage}
              className="absolute top-2 right-2 bg-white/10 hover:bg-white/20 text-white rounded-full p-2"
            >
              <X className="h-5 w-5" />
            </button>

            <img
              src={expandedSrc}
              alt="Expanded"
              className="w-full h-full object-contain rounded-lg"
            />
          </div>
        </div>
      )}
    </main>
  );
}