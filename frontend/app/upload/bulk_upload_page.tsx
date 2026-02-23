"use client";

import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { getAccessToken } from "@/lib/api";
import {
  Upload,
  FileArchive,
  Image as ImageIcon,
  MessageSquare,
  FileText,
  X,
  Check,
  AlertCircle,
  Loader2,
  Info,
} from "lucide-react";

type UploadMode = "multi-image" | "archive" | "captions" | "instructions";

interface UploadResult {
  success: boolean;
  message: string;
  total_files: number;
  successful: number;
  failed: number;
  image_ids: number[];
  errors: string[];
}

interface ValidationResult {
  valid: boolean;
  total_files: number;
  valid_images: number;
  image_list: string[];
  errors: string[];
  metadata_entries?: number;
  metadata_preview?: any[];
}

const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"];
const ALLOWED_ARCHIVE_TYPES = [".zip", ".tar", ".tar.gz", ".tgz"];
const MAX_IMAGES = 50;

export default function BulkUploadPage() {
  const [mode, setMode] = useState<UploadMode>("multi-image");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [archiveFile, setArchiveFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [isPublic, setIsPublic] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const archiveInputRef = useRef<HTMLInputElement>(null);

  const getAuthHeaders = (): Record<string, string> => {
    const token = getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  // Handle multi-image selection
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const validFiles = files.filter(f => ALLOWED_IMAGE_TYPES.includes(f.type));
    
    if (validFiles.length + selectedFiles.length > MAX_IMAGES) {
      alert(`Maximum ${MAX_IMAGES} images allowed`);
      return;
    }
    
    setSelectedFiles(prev => [...prev, ...validFiles]);
    setUploadResult(null);
  };

  // Handle archive selection
  const handleArchiveSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const isValidArchive = ALLOWED_ARCHIVE_TYPES.some(ext => 
      file.name.toLowerCase().endsWith(ext)
    );

    if (!isValidArchive) {
      alert("Invalid archive format. Allowed: .zip, .tar, .tar.gz, .tgz");
      return;
    }

    setArchiveFile(file);
    setUploadResult(null);
    setValidationResult(null);

    // Auto-validate
    await validateArchive(file);
  };

  // Drag and drop handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);

    if (mode === "multi-image") {
      const validFiles = files.filter(f => ALLOWED_IMAGE_TYPES.includes(f.type));
      if (validFiles.length + selectedFiles.length > MAX_IMAGES) {
        alert(`Maximum ${MAX_IMAGES} images allowed`);
        return;
      }
      setSelectedFiles(prev => [...prev, ...validFiles]);
    } else {
      const archiveFile = files.find(f => 
        ALLOWED_ARCHIVE_TYPES.some(ext => f.name.toLowerCase().endsWith(ext))
      );
      if (archiveFile) {
        setArchiveFile(archiveFile);
        validateArchive(archiveFile);
      }
    }
  }, [mode, selectedFiles.length]);

  // Remove selected file
  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Validate archive
  const validateArchive = async (file: File) => {
    setIsValidating(true);
    
    const formatType = mode === "captions" ? "image-caption" : 
                       mode === "instructions" ? "image-instruction" : "image-archive";
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("format_type", formatType);

    try {
      const response = await fetch("/bulk/validate-archive", {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData,
      });

      const result = await response.json();
      setValidationResult(result);
    } catch (error) {
      setValidationResult({
        valid: false,
        total_files: 0,
        valid_images: 0,
        image_list: [],
        errors: [String(error)],
      });
    } finally {
      setIsValidating(false);
    }
  };

  // Upload files
  const handleUpload = async () => {
    setIsUploading(true);
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append("is_public", String(isPublic));

      let endpoint = "/bulk/images";

      if (mode === "multi-image") {
        selectedFiles.forEach(file => {
          formData.append("files", file);
        });
      } else {
        if (!archiveFile) {
          alert("No archive selected");
          return;
        }
        formData.append("file", archiveFile);
        
        if (mode === "archive") endpoint = "/bulk/archive";
        else if (mode === "captions") endpoint = "/bulk/captions";
        else if (mode === "instructions") endpoint = "/bulk/instructions";
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData,
      });

      const result = await response.json();
      setUploadResult(result);

      if (result.success) {
        setSelectedFiles([]);
        setArchiveFile(null);
        setValidationResult(null);
      }
    } catch (error) {
      setUploadResult({
        success: false,
        message: String(error),
        total_files: 0,
        successful: 0,
        failed: 0,
        image_ids: [],
        errors: [String(error)],
      });
    } finally {
      setIsUploading(false);
    }
  };

  // Clear all
  const clearAll = () => {
    setSelectedFiles([]);
    setArchiveFile(null);
    setUploadResult(null);
    setValidationResult(null);
  };

  return (
    <main className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Bulk Upload</h1>
        <p className="text-muted-foreground mt-1">
          Upload multiple images at once
        </p>
      </div>

      {/* Mode Selection */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <button
          onClick={() => { setMode("multi-image"); clearAll(); }}
          className={cn(
            "p-4 rounded-lg border-2 transition-all text-left",
            mode === "multi-image" 
              ? "border-primary bg-primary/5" 
              : "border-border hover:border-primary/50"
          )}
        >
          <ImageIcon className="h-6 w-6 mb-2 text-primary" />
          <div className="font-medium">Multi-Image</div>
          <div className="text-xs text-muted-foreground">Up to 50 images</div>
        </button>

        <button
          onClick={() => { setMode("archive"); clearAll(); }}
          className={cn(
            "p-4 rounded-lg border-2 transition-all text-left",
            mode === "archive" 
              ? "border-primary bg-primary/5" 
              : "border-border hover:border-primary/50"
          )}
        >
          <FileArchive className="h-6 w-6 mb-2 text-primary" />
          <div className="font-medium">Archive</div>
          <div className="text-xs text-muted-foreground">Zip/tar.gz of images</div>
        </button>

        <button
          onClick={() => { setMode("captions"); clearAll(); }}
          className={cn(
            "p-4 rounded-lg border-2 transition-all text-left",
            mode === "captions" 
              ? "border-primary bg-primary/5" 
              : "border-border hover:border-primary/50"
          )}
        >
          <MessageSquare className="h-6 w-6 mb-2 text-primary" />
          <div className="font-medium">With Captions</div>
          <div className="text-xs text-muted-foreground">Images + captions.json</div>
        </button>

        <button
          onClick={() => { setMode("instructions"); clearAll(); }}
          className={cn(
            "p-4 rounded-lg border-2 transition-all text-left",
            mode === "instructions" 
              ? "border-primary bg-primary/5" 
              : "border-border hover:border-primary/50"
          )}
        >
          <FileText className="h-6 w-6 mb-2 text-primary" />
          <div className="font-medium">With Instructions</div>
          <div className="text-xs text-muted-foreground">Images + instructions.json</div>
        </button>
      </div>

      {/* Format Info */}
      {(mode === "captions" || mode === "instructions") && (
        <Card className="mb-6 bg-blue-50 border-blue-200">
          <CardContent className="pt-4">
            <div className="flex gap-3">
              <Info className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                {mode === "captions" ? (
                  <>
                    <p className="font-medium text-blue-900">Expected format: captions.json</p>
                    <pre className="mt-2 p-2 bg-white rounded text-xs overflow-x-auto">
{`[
  {"filename": "image1.jpg", "caption": "A cat sitting on a couch"},
  {"filename": "folder/image2.png", "caption": "A sunset over the ocean"}
]`}
                    </pre>
                  </>
                ) : (
                  <>
                    <p className="font-medium text-blue-900">Expected format: instructions.json</p>
                    <pre className="mt-2 p-2 bg-white rounded text-xs overflow-x-auto">
{`[
  {
    "filename": "chart1.png",
    "instruction": "What is the total revenue shown?",
    "response": "Optional ground truth answer"
  }
]`}
                    </pre>
                  </>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Drop Zone */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
              dragActive 
                ? "border-primary bg-primary/5" 
                : "border-border hover:border-primary/50"
            )}
          >
            <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-lg font-medium mb-2">
              {mode === "multi-image" 
                ? "Drop images here or click to select" 
                : "Drop archive here or click to select"}
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              {mode === "multi-image"
                ? "JPG, PNG, WebP, GIF, BMP, TIFF (max 50 files)"
                : "ZIP, TAR, TAR.GZ, TGZ (max 500MB)"}
            </p>
            
            <input
              ref={fileInputRef}
              type="file"
              multiple={mode === "multi-image"}
              accept={mode === "multi-image" ? "image/*" : ".zip,.tar,.tar.gz,.tgz"}
              onChange={mode === "multi-image" ? handleImageSelect : handleArchiveSelect}
              className="hidden"
            />
            
            <Button 
              variant="outline" 
              onClick={() => fileInputRef.current?.click()}
            >
              Select {mode === "multi-image" ? "Images" : "Archive"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Selected Files Preview */}
      {mode === "multi-image" && selectedFiles.length > 0 && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-lg">
              Selected Images ({selectedFiles.length}/{MAX_IMAGES})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
              {selectedFiles.map((file, index) => (
                <div key={index} className="relative group aspect-square">
                  <img
                    src={URL.createObjectURL(file)}
                    alt={file.name}
                    className="w-full h-full object-cover rounded"
                  />
                  <button
                    onClick={() => removeFile(index)}
                    className="absolute -top-1 -right-1 p-1 bg-destructive text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Archive Preview */}
      {mode !== "multi-image" && archiveFile && (
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <FileArchive className="h-5 w-5" />
                {archiveFile.name}
              </CardTitle>
              <Button variant="ghost" size="sm" onClick={() => { setArchiveFile(null); setValidationResult(null); }}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <CardDescription>
              {(archiveFile.size / 1024 / 1024).toFixed(2)} MB
            </CardDescription>
          </CardHeader>
          
          {isValidating && (
            <CardContent>
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Validating archive...
              </div>
            </CardContent>
          )}

          {validationResult && (
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  {validationResult.valid ? (
                    <Badge variant="success" className="gap-1">
                      <Check className="h-3 w-3" /> Valid
                    </Badge>
                  ) : (
                    <Badge variant="destructive" className="gap-1">
                      <AlertCircle className="h-3 w-3" /> Invalid
                    </Badge>
                  )}
                  <span className="text-sm text-muted-foreground">
                    {validationResult.valid_images} valid images found
                  </span>
                </div>

                {validationResult.metadata_entries !== undefined && (
                  <div className="text-sm text-muted-foreground">
                    {validationResult.metadata_entries} metadata entries found
                  </div>
                )}

                {validationResult.errors.length > 0 && (
                  <div className="bg-destructive/10 rounded p-3">
                    <p className="text-sm font-medium text-destructive mb-1">Errors:</p>
                    <ul className="text-xs text-destructive space-y-1">
                      {validationResult.errors.slice(0, 5).map((error, i) => (
                        <li key={i}>• {error}</li>
                      ))}
                      {validationResult.errors.length > 5 && (
                        <li>...and {validationResult.errors.length - 5} more</li>
                      )}
                    </ul>
                  </div>
                )}

                {validationResult.image_list.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-1">Preview:</p>
                    <div className="text-xs text-muted-foreground space-y-0.5 max-h-32 overflow-y-auto">
                      {validationResult.image_list.map((img, i) => (
                        <div key={i}>• {img}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          )}
        </Card>
      )}

      {/* Upload Result */}
      {uploadResult && (
        <Card className={cn(
          "mb-6",
          uploadResult.success ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"
        )}>
          <CardContent className="pt-4">
            <div className="flex items-start gap-3">
              {uploadResult.success ? (
                <Check className="h-5 w-5 text-green-600 flex-shrink-0" />
              ) : (
                <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0" />
              )}
              <div>
                <p className={cn(
                  "font-medium",
                  uploadResult.success ? "text-green-900" : "text-red-900"
                )}>
                  {uploadResult.message}
                </p>
                <p className="text-sm mt-1">
                  {uploadResult.successful} of {uploadResult.total_files} files uploaded successfully
                </p>
                {uploadResult.errors.length > 0 && (
                  <ul className="text-xs mt-2 space-y-1">
                    {uploadResult.errors.slice(0, 5).map((error, i) => (
                      <li key={i} className="text-red-700">• {error}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Upload Controls */}
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={isPublic}
            onChange={(e) => setIsPublic(e.target.checked)}
            className="rounded border-border"
          />
          <span className="text-sm">Make images public</span>
        </label>

        <div className="flex gap-3">
          <Button variant="outline" onClick={clearAll}>
            Clear All
          </Button>
          <Button
            onClick={handleUpload}
            disabled={
              isUploading || 
              (mode === "multi-image" ? selectedFiles.length === 0 : !archiveFile) ||
              (mode !== "multi-image" && validationResult && !validationResult.valid)
            }
          >
            {isUploading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="h-4 w-4 mr-2" />
                Upload {mode === "multi-image" ? `${selectedFiles.length} Images` : "Archive"}
              </>
            )}
          </Button>
        </div>
      </div>
    </main>
  );
}
