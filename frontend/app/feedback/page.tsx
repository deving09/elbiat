"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import {
  MessageSquare,
  Check,
  X,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  Loader2,
  Star,
  Pencil,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getAccessToken } from "@/lib/api";

interface Convo {
  id: number;
  image_id: number;
  feedback: string;
  enabled: boolean;
  created_at: string;
  prompt: string;
  response: string;
  feedback_length: number;
  attribution_score: number;
  model_name: string;
  task: string;
}

interface FeedbackListResponse {
  items: Convo[];
  total: number;
  page: number;
  page_size: number;
}

const fetchFeedback = async (
  page: number,
  pageSize: number,
  sortBy: string,
  sortOrder: string
): Promise<FeedbackListResponse> => {
  const token = getAccessToken();
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
    sort_by: sortBy,
    sort_order: sortOrder,
  });
  const response = await fetch(`/api/feedback?${params}`, {
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

export default function FeedbackPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
  const [sortBy, setSortBy] = useState<string>("created_at");
  const [sortOrder, setSortOrder] = useState<string>("desc");
  
  // Card detail view
  const [selectedConvo, setSelectedConvo] = useState<Convo | null>(null);
  
  // Image lightbox
  const [lightboxOpen, setLightboxOpen] = useState(false);
  
  // Feedback editing
  const [isEditingFeedback, setIsEditingFeedback] = useState(false);
  const [editedFeedback, setEditedFeedback] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["feedback", page, pageSize, sortBy, sortOrder],
    queryFn: () => fetchFeedback(page, pageSize, sortBy, sortOrder),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: { feedback?: string; enabled?: boolean } }) =>
      updateFeedback(id, data),
    onSuccess: (updatedConvo) => {
      queryClient.invalidateQueries({ queryKey: ["feedback"] });
      if (selectedConvo) {
        setSelectedConvo(updatedConvo);
      }
    },
  });

  const openCard = (convo: Convo) => {
    setSelectedConvo(convo);
    setIsEditingFeedback(false);
    setEditedFeedback(convo.feedback);
  };

  const closeCard = () => {
    setSelectedConvo(null);
    setIsEditingFeedback(false);
    setEditedFeedback("");
  };

  const startEditingFeedback = () => {
    if (selectedConvo) {
      setEditedFeedback(selectedConvo.feedback);
      setIsEditingFeedback(true);
    }
  };

  const cancelEditingFeedback = () => {
    setIsEditingFeedback(false);
    if (selectedConvo) {
      setEditedFeedback(selectedConvo.feedback);
    }
  };

  const saveFeedback = () => {
    if (selectedConvo && editedFeedback !== selectedConvo.feedback) {
      updateMutation.mutate({
        id: selectedConvo.id,
        data: { feedback: editedFeedback },
      });
    }
    setIsEditingFeedback(false);
  };

  const toggleEnabled = (convo: Convo) => {
    updateMutation.mutate({
      id: convo.id,
      data: { enabled: !convo.enabled },
    });
  };

  const toggleSort = () => {
    setSortOrder(sortOrder === "desc" ? "asc" : "desc");
    setPage(1);
  };

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  const openDialog = (convo: Convo) => {
    setSelectedConvo(convo);
    setEditedFeedback(convo.feedback);
  };
  
  const closeDialog = () => {
    setSelectedConvo(null);
    setEditedFeedback("");
  };
 

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Feedback</h1>
          <p className="text-muted-foreground mt-1">
            View and manage your feedback contributions
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Sort by:</span>
          <Select value={sortBy} onValueChange={(v) => { setSortBy(v); setPage(1); }}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="created_at">Date</SelectItem>
              <SelectItem value="feedback_length">Feedback Length</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={toggleSort}>
            <ArrowUpDown className={cn("h-4 w-4 transition-transform", sortOrder === "asc" && "rotate-180")} />
          </Button>
        </div>
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
            {data.items.map((convo) => (
              <Card
                key={convo.id}
                className={cn(
                  "group overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary transition-all",
                  !convo.enabled && "opacity-50"
                )}
                onClick={() => openCard(convo)}
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
                    <Badge variant="secondary" className="text-xs">
                      <Star className="h-3 w-3 mr-1" />
                      {convo.attribution_score.toFixed(2)}
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
                  <p className="text-xs text-muted-foreground mt-1">
                    {new Date(convo.created_at).toLocaleDateString()}
                  </p>
                </CardContent>
              </Card>
            ))}
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

      {/* Detail Dialog */}
        <Dialog open={!!selectedConvo} onOpenChange={() => closeDialog()}>
        {selectedConvo && (
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                Feedback #{selectedConvo.id}
                <Badge variant={selectedConvo.enabled ? "default" : "secondary"}>
                    {selectedConvo.enabled ? "Enabled" : "Disabled"}
                </Badge>
                </DialogTitle>
            </DialogHeader>

            <div className="grid md:grid-cols-2 gap-6 mt-4">
                {/* Image */}
                <div>
                <img
                    src={`/images/${selectedConvo.image_id}/file`}
                    alt={`Image ${selectedConvo.image_id}`}
                    className="w-full rounded-lg"
                />
                <div className="mt-2 flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">Attribution Score:</span>
                    <Badge><Star className="h-3 w-3 mr-1" />{selectedConvo.attribution_score.toFixed(4)}</Badge>
                </div>
                </div>

                {/* Content */}
                <div className="space-y-4">
                <div>
                    <label className="text-sm font-medium">Prompt</label>
                    <div className="mt-1 p-3 bg-muted rounded-lg text-sm">{selectedConvo.prompt}</div>
                </div>

                <div>
                    <label className="text-sm font-medium">Model Response</label>
                    <div className="mt-1 p-3 bg-muted rounded-lg text-sm max-h-40 overflow-y-auto">
                    {selectedConvo.response}
                    </div>
                </div>

                <div>
                    <label className="text-sm font-medium">Your Feedback</label>
                    <Textarea
                    value={editedFeedback}
                    onChange={(e) => setEditedFeedback(e.target.value)}
                    className="mt-1"
                    rows={4}
                    />
                </div>

                <div className="flex items-center justify-between pt-4">
                    <Button
                    variant="outline"
                    onClick={() => {
                        updateMutation.mutate({
                        id: selectedConvo.id,
                        data: { enabled: !selectedConvo.enabled },
                        });
                        setSelectedConvo({ ...selectedConvo, enabled: !selectedConvo.enabled });
                    }}
                    >
                    {selectedConvo.enabled ? <><X className="h-4 w-4 mr-2" />Disable</> : <><Check className="h-4 w-4 mr-2" />Enable</>}
                    </Button>

                    <div className="flex gap-2">
                    <Button variant="outline" onClick={closeDialog}>Cancel</Button>
                    <Button onClick={saveFeedback} disabled={updateMutation.isPending}>
                        {updateMutation.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Check className="h-4 w-4 mr-2" />}
                        Save
                    </Button>
                    </div>
                </div>

                <div className="text-xs text-muted-foreground pt-2">
                    Model: {selectedConvo.model_name} • Task: {selectedConvo.task} • {new Date(selectedConvo.created_at).toLocaleString()}
                </div>
                </div>
            </div>
            </DialogContent>
        )}
        </Dialog>

      {/* Image Lightbox */}
      <Dialog open={lightboxOpen} onOpenChange={setLightboxOpen}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] p-2 bg-black/90">
          {selectedConvo && (
            <div className="flex items-center justify-center h-full">
              <img
                src={`/images/${selectedConvo.image_id}/file`}
                alt={`Image ${selectedConvo.image_id}`}
                className="max-w-full max-h-[90vh] object-contain"
              />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </main>
  );
}