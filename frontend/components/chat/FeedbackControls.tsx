"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { api } from "@/lib/api";
import { getFeedbackConfig } from "@/lib/feedbackConfig";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface FeedbackControlsProps {
  chatSessionId?: number;  // Optional - will be created on first feedback
  turnIndex: number;
  messageContent: string;
  messages: ChatMessage[];  // Add this - full chat context
  onFeedbackSubmit?: (chatSessionId: number) => void;  // Returns chat_session_id
}

export function FeedbackControls({
  chatSessionId,
  turnIndex,
  messageContent,
  messages,
  onFeedbackSubmit,
}: FeedbackControlsProps) {
  const { user } = useAuth();
  const [thumbs, setThumbs] = useState<"up" | "down" | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(messageContent);
  const [showCommentInput, setShowCommentInput] = useState(false);
  const [commentText, setCommentText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<number | undefined>(chatSessionId);
  if (!user) return null;

  const config = getFeedbackConfig(user.experiment_bucket);

  const handleThumbsClick = async (value: "up" | "down") => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    try {
      const result = await api.submitFeedback({
        chat_session_id: currentSessionId,
        messages: currentSessionId ? undefined : messages,
        turn_index: turnIndex,
        thumbs: value,
      });
      setThumbs(value);
      setCurrentSessionId(result.chat_session_id);
      onFeedbackSubmit?.(result.chat_session_id);
    } catch (error) {
      console.error("Failed to submit thumbs feedback:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleEditSubmit = async () => {
    if (isSubmitting || editedContent === messageContent) return;
    setIsSubmitting(true);
    try {
      const result = await api.submitFeedback({
        chat_session_id: currentSessionId,
        messages: currentSessionId ? undefined : messages,
        turn_index: turnIndex,
        edit_original: messageContent,
        edit_revised: editedContent,
      });
      setIsEditing(false);
      setCurrentSessionId(result.chat_session_id);
      onFeedbackSubmit?.(result.chat_session_id);
    } catch (error) {
      console.error("Failed to submit edit feedback:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCommentSubmit = async () => {
    if (isSubmitting || !commentText.trim()) return;
    setIsSubmitting(true);
    try {
      const result = await api.submitFeedback({
        chat_session_id: currentSessionId,
        messages: currentSessionId ? undefined : messages,
        turn_index: turnIndex,
        comment_text: commentText.trim(),
      });
      setCommentText("");
      setShowCommentInput(false);
      setCurrentSessionId(result.chat_session_id);
      onFeedbackSubmit?.(result.chat_session_id);
    } catch (error) {
      console.error("Failed to submit comment:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-2 mt-2">
      {config.features.showThumbs && (
        <div className="flex gap-2">
          <button
            onClick={() => handleThumbsClick("up")}
            disabled={isSubmitting}
            className={`p-1 rounded transition-colors ${
              thumbs === "up"
                ? "bg-green-100 text-green-600"
                : "hover:bg-gray-100 text-gray-400"
            }`}
            title="Good response"
          >
            <ThumbsUpIcon />
          </button>
          <button
            onClick={() => handleThumbsClick("down")}
            disabled={isSubmitting}
            className={`p-1 rounded transition-colors ${
              thumbs === "down"
                ? "bg-red-100 text-red-600"
                : "hover:bg-gray-100 text-gray-400"
            }`}
            title="Bad response"
          >
            <ThumbsDownIcon />
          </button>

          {config.features.showEdit && (
            <button
              onClick={() => setIsEditing(!isEditing)}
              disabled={isSubmitting}
              className={`p-1 rounded transition-colors ${
                isEditing ? "bg-blue-100 text-blue-600" : "hover:bg-gray-100 text-gray-400"
              }`}
              title="Suggest an edit"
            >
              <EditIcon />
            </button>
          )}

          {config.features.showComment && (
            <button
              onClick={() => setShowCommentInput(!showCommentInput)}
              disabled={isSubmitting}
              className={`p-1 rounded transition-colors ${
                showCommentInput ? "bg-purple-100 text-purple-600" : "hover:bg-gray-100 text-gray-400"
              }`}
              title="Add a comment"
            >
              <CommentIcon />
            </button>
          )}
        </div>
      )}

      {isEditing && (
        <div className="border rounded-lg p-3 bg-gray-50">
          <p className="text-xs text-gray-500 mb-2">Suggest a better response:</p>
          <textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            className="w-full p-2 border rounded text-sm min-h-[100px] text-gray-900 bg-white"
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={() => { setIsEditing(false); setEditedContent(messageContent); }}
              className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-200 rounded"
            >
              Cancel
            </button>
            <button
              onClick={handleEditSubmit}
              disabled={isSubmitting || editedContent === messageContent}
              className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              Submit Edit
            </button>
          </div>
        </div>
      )}

      {showCommentInput && (
        <div className="border rounded-lg p-3 bg-gray-50">
          <p className="text-xs text-gray-500 mb-2">Add a comment:</p>
          <textarea
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
            className="w-full p-2 border rounded text-sm min-h-[60px] text-gray-900 bg-white"
            placeholder="What could be improved?"
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={() => { setShowCommentInput(false); setCommentText(""); }}
              className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-200 rounded"
            >
              Cancel
            </button>
            <button
              onClick={handleCommentSubmit}
              disabled={isSubmitting || !commentText.trim()}
              className="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
            >
              Submit Comment
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Icons
function ThumbsUpIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
    </svg>
  );
}

function ThumbsDownIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
    </svg>
  );
}

function EditIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
    </svg>
  );
}

function CommentIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
    </svg>
  );
}