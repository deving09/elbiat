from sqlalchemy import String, Integer, Text, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import VARCHAR


from .db import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)



class Convo(Base):
    __tablename__ = "convos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image: Mapped[str] = mapped_column(String, nullable=False) #filename

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

    user_id: Mapped[int] = mapped_column(Integer, nullable=False)



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
        nullable=False,
        index=True,
    )

    # Where you stored it locally / in blob storage
    image_path: Mapped[str] = mapped_column(
        VARCHAR,
        nullable=False,
        unique=True,
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


