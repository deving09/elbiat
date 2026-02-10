from sqlalchemy import String, Integer, Text, Boolean, func, DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import VARCHAR
from sqlalchemy import Enum, String
import enum


from datetime import datetime


from .db import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)



class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    
    display_name: Mapped[str] = mapped_column(String)

    vlmeval_data: Mapped[str] = mapped_column(String, index=True)

    description: Mapped[str] = mapped_column(String)

    primary_metric: Mapped[str] = mapped_column(String)

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


