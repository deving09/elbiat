from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import List, Dict, Optional

from .models import EvalStatus, AdaptationLevel, AdaptationStatus

from datetime import datetime



class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    email: EmailStr
    role: str
    experiment_bucket: Optional[str] = None
    is_active: bool
    created_at: datetime


class ConvoCreate(BaseModel):
    #user_id: int
    image_id: int
    conversations: List[Dict]
    model_name: str
    model_type: str
    task: str
    feedback: str
    monetized: Optional[bool] = True
    enabled: Optional[bool] = True


class ImageCreate(BaseModel):
    sha256: str
    phash: str
    image_url: str
    image_path: str
    content_length: int


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    is_public: bool
    created_at: Optional[str] = None  # or datetime if you use it


class ImgHashCheck(BaseModel):
    sha256: Optional[str] = None
    phash: Optional[str] = None
    content_length: int


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)

class SignupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    email: EmailStr
    role: str
    experiment_bucket: Optional[str] = None


class TaskCreate(BaseModel):
    name: str
    vlmeval_data: str
    description: str
    primary_metric_type: str
    primary_metric_key: str
    display_name : Optional[str] = None
    primary_metric_suffix: Optional[str] = None
    num_examples: Optional[int] = None
    paper_url: Optional[str] = None
    dataset_url: Optional[str] = None
    dataset_version: Optional[str] = None

class TaskResponse(TaskCreate):
    id: int
    created_at: datetime

    #class Config:
    #    from_attributes = True


class ModelRegister(BaseModel):
    name: str
    display_name : str
    vlmeval_model: str
    default_args: Optional[List[Dict]]
    model_type: str = "vlm"


class ModelResponse(ModelRegister):
    id: int

class CreateEvalRun(BaseModel):
    task_id: int
    model_id: int
    status: EvalStatus
    metrics: Optional[dict] = None
    artifacts_dir: Optional[str] = None
    command: Optional[str] = None
    git_commit: Optional[str] = None


class EvalRunResponse(CreateEvalRun):
    id: int
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]  = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None         

    # Joined fields
    task_name: Optional[str] = None
    model_name: Optional[str] = None
    model_display_name: Optional[str] = None
    primary_metric: Optional[float] = None



class LeaderboardEntry(BaseModel):
    model_name: str
    model_display_name: str
    primary_metric: Optional[float]
    run_id: int
    run_date: datetime
    #git_commit: Optional[str] = None
    status: str

# ─────────────────────────────────────────────────────────────────────────────
# Chat Feedback schemas (experiment-based)
# ─────────────────────────────────────────────────────────────────────────────

class ChatFeedbackCreate(BaseModel):
    convo_id: int
    turn_index: int
    thumbs: Optional[str] = None  # "up" or "down"
    edit_original: Optional[str] = None
    edit_revised: Optional[str] = None


class ChatFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    convo_id: int
    user_id: int
    turn_index: int
    thumbs: Optional[str] = None
    edit_original: Optional[str] = None
    edit_revised: Optional[str] = None
    experiment_bucket: Optional[str] = None
    available_actions: Optional[list[str]] = None
    created_at: datetime


class ChatCommentCreate(BaseModel):
    convo_id: int
    turn_index: int
    text: str


class ChatCommentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    convo_id: int
    user_id: int
    turn_index: int
    text: str
    experiment_bucket: Optional[str] = None
    created_at: datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class FeedbackSubmit(BaseModel):
    """Submit feedback, creating a ChatSession if needed."""
    chat_session_id: Optional[int] = None  # None = create new session
    messages: Optional[List[ChatMessage]] = None  # Required if chat_session_id is None
    title: Optional[str] = None  # Optional title for new session
    turn_index: int
    thumbs: Optional[str] = None  # "up" or "down"
    edit_original: Optional[str] = None
    edit_revised: Optional[str] = None
    comment_text: Optional[str] = None  # Optional comment in same submission


class FeedbackSubmitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chat_session_id: int
    feedback_id: int
    comment_id: Optional[int] = None


class QuestionOut(BaseModel):
    id: int
    image_id: int
    image_path: str
    question_text: str

class AnswerCreate(BaseModel):
    answer_text: str


# ─────────────────────────────────────────────────────────────────────────────
# User Adaptation schemas
# ─────────────────────────────────────────────────────────────────────────────

class TriggerAdaptationRequest(BaseModel):
    """Request to create/trigger a user adaptation."""
    level: AdaptationLevel
    base_model_id: int
    config: Optional[Dict] = None  # Method-specific config


class UserAdaptationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    base_model_id: int
    level: str
    status: str
    artifact_path: Optional[str] = None
    config: Optional[Dict] = None
    metrics: Optional[Dict] = None
    contribution_count: int
    created_at: datetime
    last_trained_at: Optional[datetime] = None


class UserAdaptationSummary(BaseModel):
    """Lightweight version for listings."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    level: str
    status: str
    contribution_count: int
    last_trained_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Image Embedding schemas
# ─────────────────────────────────────────────────────────────────────────────

class ImageEmbeddingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    model_name: str
    created_at: datetime
    # Note: embedding itself usually too large to return in API


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison schemas
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonRequest(BaseModel):
    """Request side-by-side inference."""
    image_id: int
    prompt: str
    adaptation_id: Optional[int] = None  # If None, use user's default/latest


class ComparisonResponse(BaseModel):
    """Side-by-side model outputs."""
    image_id: int
    prompt: str
    base_response: str
    adapted_response: str
    adaptation_level: str
    adaptation_id: int


class ComparisonFeedbackCreate(BaseModel):
    """Feedback on a side-by-side comparison."""
    image_id: int
    adaptation_id: int
    prompt: str
    base_response: str
    adapted_response: str
    
    # Preference: which was better?
    preference: Optional[str] = None  # "base", "adapted", "tie"
    
    # Per-response feedback (optional)
    base_thumbs: Optional[str] = None  # "up" / "down"
    adapted_thumbs: Optional[str] = None
    
    # Edits (optional)
    base_edit: Optional[str] = None
    adapted_edit: Optional[str] = None
    
    # Free-form comment
    comment: Optional[str] = None


class ComparisonFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int
    image_id: int
    adaptation_id: int
    prompt: str
    base_response: Optional[str] = None 
    adapted_response: Optional[str] = None
    preference: Optional[str] = None
    base_thumbs: Optional[str] = None
    adapted_thumbs: Optional[str] = None
    created_at: datetime