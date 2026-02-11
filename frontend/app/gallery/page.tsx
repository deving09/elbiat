"use client";

import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type Image } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Image as ImageIcon,
  Upload,
  Globe,
  Lock,
  Loader2,
  Eye,
  MoreVertical,
} from "lucide-react";

export default function GalleryPage() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadingPublic, setUploadingPublic] = useState(false);
  const [filter, setFilter] = useState<"all" | "public" | "private">("all");

  const { data: images, isLoading } = useQuery({
    queryKey: ["images"],
    queryFn: () => api.getImages(),
  });

  const uploadMutation = useMutation({
    mutationFn: ({ file, isPublic }: { file: File; isPublic: boolean }) =>
      api.uploadImage(file, isPublic),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["images"] });
    },
  });

  const toggleVisibilityMutation = useMutation({
    mutationFn: ({ id, isPublic }: { id: number; isPublic: boolean }) =>
      api.toggleImageVisibility(id, isPublic),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["images"] });
    },
  });

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadMutation.mutateAsync({ file, isPublic: uploadingPublic });
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const filteredImages = images?.filter((img) => {
    if (filter === "all") return true;
    if (filter === "public") return img.is_public;
    if (filter === "private") return !img.is_public;
    return true;
  });

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Gallery</h1>
          <p className="text-muted-foreground mt-1">
            Browse and manage your images
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="image/*"
            className="hidden"
          />
          <Button
            variant="outline"
            onClick={() => {
              setUploadingPublic(false);
              fileInputRef.current?.click();
            }}
            disabled={uploadMutation.isPending}
          >
            {uploadMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Lock className="h-4 w-4 mr-2" />
            )}
            Upload Private
          </Button>
          <Button
            onClick={() => {
              setUploadingPublic(true);
              fileInputRef.current?.click();
            }}
            disabled={uploadMutation.isPending}
          >
            {uploadMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Globe className="h-4 w-4 mr-2" />
            )}
            Upload Public
          </Button>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex items-center space-x-2 mb-6">
        <Button
          variant={filter === "all" ? "default" : "outline"}
          size="sm"
          onClick={() => setFilter("all")}
        >
          All
        </Button>
        <Button
          variant={filter === "public" ? "default" : "outline"}
          size="sm"
          onClick={() => setFilter("public")}
        >
          <Globe className="h-4 w-4 mr-2" />
          Public
        </Button>
        <Button
          variant={filter === "private" ? "default" : "outline"}
          size="sm"
          onClick={() => setFilter("private")}
        >
          <Lock className="h-4 w-4 mr-2" />
          Private
        </Button>
      </div>

      {/* Images grid */}
      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div
              key={i}
              className="aspect-square bg-muted rounded-xl animate-pulse"
            />
          ))}
        </div>
      ) : filteredImages && filteredImages.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          {filteredImages.map((image) => (
            <Card
              key={image.id}
              className="group overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary transition-all"
            >
              <div className="relative aspect-square">
                <img
                  //src={image.url}
                  src={`${process.env.NEXT_PUBLIC_API_BASE ?? ""}/images/${image.id}/file`}
                  alt={image.filename}
                  className="w-full h-full object-cover"
                />
                {/* Overlay */}
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleVisibilityMutation.mutate({
                        id: image.id,
                        isPublic: !image.is_public,
                      });
                    }}
                  >
                    {image.is_public ? (
                      <>
                        <Lock className="h-4 w-4 mr-1" />
                        Make Private
                      </>
                    ) : (
                      <>
                        <Globe className="h-4 w-4 mr-1" />
                        Make Public
                      </>
                    )}
                  </Button>
                </div>
                {/* Visibility badge */}
                <div className="absolute top-2 right-2">
                  <Badge
                    variant={image.is_public ? "default" : "secondary"}
                    className="text-xs"
                  >
                    {image.is_public ? (
                      <Globe className="h-3 w-3 mr-1" />
                    ) : (
                      <Lock className="h-3 w-3 mr-1" />
                    )}
                    {image.is_public ? "Public" : "Private"}
                  </Badge>
                </div>
              </div>
              <CardContent className="p-3">
                <p className="text-sm truncate">{image.filename}</p>
                <p className="text-xs text-muted-foreground">
                  {new Date(image.created_at).toLocaleDateString()}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <ImageIcon className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">No images yet</p>
            <p className="text-muted-foreground">
              Upload an image to get started
            </p>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
