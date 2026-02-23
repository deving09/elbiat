from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import List, Dict, Optional

from .models import EvalStatus

from datetime import datetime

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
    id: int
    email: EmailStr


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