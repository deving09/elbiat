from sqlalchemy import String, Integer, Text, Boolean, func, DateTime, Float, UniqueConstraint
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import VARCHAR
from sqlalchemy.sql import expression
from sqlalchemy import Enum, String
import enum

from typing import Optional


from datetime import datetime


from .db import Base


# Add this enum near the top (after the imports, before User class)
class UserRole(str, enum.Enum):
    USER = "user"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"



class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # New fields
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole),
        nullable=False,
        default=UserRole.USER,
        server_default="user",
    )
    experiment_bucket: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )



class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    
    display_name: Mapped[str] = mapped_column(String)

    vlmeval_data: Mapped[str] = mapped_column(String, index=True)

    description: Mapped[str] = mapped_column(String)

    #primary_metric: Mapped[str] = mapped_column(String)

     # 🔁 Renamed
    primary_metric_type: Mapped[str] = mapped_column(String)

    # ➕ New field
    primary_metric_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="avg",                     # ORM default
        server_default="avg",              # DB default
    )


    primary_metric_suffix: Mapped[str] = mapped_column(String)

    num_examples: Mapped[int] = mapped_column(Integer, nullable=True)

    paper_url: Mapped[str] = mapped_column(String, nullable=True)

    dataset_url: Mapped[str] = mapped_column(String, nullable=True)

    dataset_version: Mapped[str] = mapped_column(String, nullable=True)

    user_id: Mapped[int] = mapped_column(Integer, 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )



class Models(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    name: Mapped[str] = mapped_column(String, unique=True)

    display_name: Mapped[str] = mapped_column(String, nullable=True)

    vlmeval_model: Mapped[str] = mapped_column(String)

    default_args: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)

    model_type: Mapped[str] = mapped_column(String, server_default="vlm")

    # Add these new columns:
    model_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_finetuned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    base_model: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)




class EvalStatus(str, enum.Enum):
    """ Enumeration for evaluation status """
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "completed"
    FAILED = "failed"

class Evals(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
 
    model_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    status: Mapped[EvalStatus] = mapped_column(Enum(EvalStatus), nullable=False, default=EvalStatus.QUEUED)

    metrics : Mapped[dict] = mapped_column(JSONB)

    artifacts_dir: Mapped[str] = mapped_column(String, nullable=True)

    command: Mapped[str] = mapped_column(String, nullable=True)

    git_commit: Mapped[str] = mapped_column(String, nullable=True)

    error: Mapped[str] = mapped_column(Text, nullable=True)
    

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=None, #func.now(),
        nullable=True,
        index=True,
    )
    

    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=None, #func.now(),
        nullable=True,
        index=True,
    )



class Convo(Base):
    __tablename__ = "convos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    image_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    conversations: Mapped[list[dict]] = mapped_column(JSONB, nullable=False)

    model_name: Mapped[str] = mapped_column(String, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    task: Mapped[str] = mapped_column(String, nullable=False, server_default="open")

    feedback: Mapped[str] = mapped_column(Text, nullable=False)


    attributions: Mapped[list["DataAttribution"]] = relationship("DataAttribution", back_populates="convo")


    monetized: Mapped[bool] = mapped_column(
            Boolean,
            nullable=False,
            server_default="true"
    )

    enabled: Mapped[bool] = mapped_column(
            Boolean,
            nullable=False,
            server_default="true"
            )

    user_id: Mapped[int] = mapped_column(Integer, 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    


class Image(Base):
    __tablename__ = "images"

    # Surrogate primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Exact content hash (raw bytes)
    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    # Perceptual hash (near-duplicate detection later)
    # 64-bit pHash → 16 hex chars
    phash: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,
    )

    # Where the image came from
    image_url: Mapped[str] = mapped_column(
        VARCHAR,  # unbounded string in Postgres
        nullable=True,
        index=True,
    )

    # Where you stored it locally / in blob storage
    image_path: Mapped[str] = mapped_column(
        VARCHAR,
        nullable=False,
        unique=True,
    )

    user_id: Mapped[int] = mapped_column(Integer, 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Byte size of downloaded content
    content_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    # Audit / ordering
    created_at: Mapped["datetime"] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


    is_public: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=expression.false(),  # DB-level default
        default=False,                      # ORM-level default
    )



class ImageCaption(Base):
    __tablename__ = "image_captions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="user_input")
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ImageInstruction(Base):
    __tablename__ = "image_instructions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, default="vqa")
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="user_input")
    
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )



class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    image_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Timing
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Optional metadata
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class DataAttribution(Base):
    """
    Stores influence scores from data attribution methods (TracIn, LogIX, etc.)
    Links training examples (convos) to their impact on benchmark performance.
    """
    __tablename__ = "data_attributions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Which training example
    convo_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("convos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Which model checkpoint was evaluated
    checkpoint_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Which benchmark (chartqa, mochi_grid, mochi_naive, blink, cvbench)
    benchmark: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Attribution method used (tracin, logix, trak, etc.)
    method: Mapped[str] = mapped_column(String(50), nullable=False, default="tracin")
    
    # The raw influence score from the method
    influence_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Normalized score (z-score within benchmark, for cross-benchmark comparison)
    z_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Rank within this benchmark (1 = most positive influence)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Optional: per-test-example breakdown or other method-specific data
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationships
    convo: Mapped["Convo"] = relationship("Convo", back_populates="attributions")
    
    __table_args__ = (
        # Unique constraint: one score per (convo, checkpoint, benchmark, method)
        UniqueConstraint("convo_id", "checkpoint_id", "benchmark", "method", 
                        name="uq_attribution_convo_checkpoint_benchmark_method"),
    )