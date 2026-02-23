import hashlib
import os
import random
import string
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi import Query
from pydantic import BaseModel, HttpUrl, ConfigDict
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from typing import List, Optional, Literal

from PIL import Image
from io import BytesIO
import numpy as np
from datetime import datetime


from app import models
from app.db import get_db
from app.deps import get_current_user

router = APIRouter(prefix="/images", tags=["images"])

IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", "images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE = 2048


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _rand_suffix(k: int = 4) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=k))


def _ext_from_content_type(content_type: Optional[str]) -> str:
    if not content_type:
        return "jpg"
    ct = content_type.split(";")[0].strip().lower()
    return {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tif",
    }.get(ct, "jpg")


def _atomic_write(path: Path, img: Image.Image) -> None:
    #tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(path)
    #os.replace(tmp, path)




def _resize_for_saving(img: Image.Image, max_image_size: int = MAX_IMAGE_SIZE):
    """ Sets the max single side size and scales the image proportionally to fit """
    o_width, o_height = img.size


    if o_width > max_image_size and o_width > o_height:
        new_width = max_image_size
        new_height = int((max_image_size * o_height) / o_width)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif o_height > max_image_size and o_height > o_width:
        new_height = max_image_size
        new_width = int((max_image_size * o_width) / o_height)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img



def _normalize_image_for_hash(img: Image.Image) -> Image.Image:
    """
    Normalize to a stable format before perceptual hash:
    - convert to RGB
    - handle weird modes (P, LA, RGBA, etc.)
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _compute_phash(img: Image.Image, hash_size: int = 8 ) -> str:
    """
    Perceptual hash (pHash) implementation using DCT.
    Returns a hex string. Good for near-duplicate detection.
    """

    img = _normalize_image_for_hash(img).convert("L")  # grayscale
    # pHash uses a larger resize than hash_size; common is 32x32
    img = img.resize((32, 32), Image.Resampling.LANCZOS)

    pixels = np.asarray(img, dtype=np.float32)

    # 2D DCT
    # implement DCT via scipy if available; otherwise use numpy FFT trick fallback
    try:
        from scipy.fftpack import dct
        dct_rows = dct(pixels, axis=0, norm="ortho")
        dct_2d = dct(dct_rows, axis=1, norm="ortho")
    except Exception:
        # fallback: slower/rougher but works without scipy
        import numpy.fft as fft
        dct_2d = fft.fft2(pixels).real

    # take top-left 8x8 (excluding DC term optionally)
    dct_lowfreq = dct_2d[:hash_size, :hash_size]
    # median excluding DC for stability
    dct_flat = dct_lowfreq.flatten()
    med = float(np.median(dct_flat[1:]))

    bits = (dct_lowfreq > med).astype(np.uint8).flatten()
    # pack bits into hex
    bit_string = "".join("1" if b else "0" for b in bits)
    return hex(int(bit_string, 2))[2:].zfill((hash_size * hash_size + 3) // 4)


def _dedupe_by_sha(db: Session, sha256: str, content_length: int) -> Optional[models.Image]:
    # Use sha256 as unique truth; content_length as sanity check
    return db.execute(
        select(models.Image).where(
            models.Image.sha256 == sha256,
            models.Image.content_length == content_length,
        )
    ).scalar_one_or_none()


#def _new_image_fn(db: Session = Depends(get_db), images_folder: str = "images"):
def _new_image_fn(db: Session) -> str:

    max_id = db.execute(
        select(func.max(models.Image.id))
    ).scalar()

    next_id = (max_id or 0) + 1

    suffix = _rand_suffix()

    filename = f"{next_id}_{suffix}"
    return filename





def _save_image_row(
    db: Session,
    *,
    user_id: int,
    sha256: str,
    phash: str,
    image_url: Optional[str],
    image_path: str,
    content_length: int,
) -> models.Image:
    row = models.Image(
        user_id=user_id,
        sha256=sha256,
        phash=phash,
        image_url=image_url,
        image_path=image_path,
        content_length=content_length,
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
        return row
    except IntegrityError:
        db.rollback()
        # another request inserted same sha256 concurrently
        existing = db.execute(
            select(models.Image).where(models.Image.sha256 == sha256)
        ).scalar_one()
        return existing


class IngestUrlRequest(BaseModel):
    #user_id: int
    image_url: HttpUrl


# ----------------------------
# Endpoint 1: ingest from URL
# ----------------------------
@router.post("/ingest_url")
#def ingest_url(payload: IngestUrlRequest, db: Session = Depends(get_db), user=Depends(get_current_user)):
def ingest_url(payload: IngestUrlRequest, db: Session = Depends(get_db), user=Depends(get_current_user)):
    # Fast path: if exact same URL already exists, return it
    user_id = user.id
    
    existing_url = db.execute(
        select(models.Image).where(models.Image.image_url == str(payload.image_url))
    ).scalar_one_or_none()
    if existing_url:
        return {
            "status": "exists_url",
            "image_id": existing_url.id,
            "image_path": existing_url.image_path,
            "sha256": existing_url.sha256,
        }
    
    # Download bytes server-side
    try:
        r = requests.get(str(payload.image_url), stream=True, timeout=20)
        r.raise_for_status()
        data = r.content  # fine for now; streaming-to-file can come later
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image_url: {e}")

    content_length = len(data)
    sha256 = _sha256_bytes(data)


    # Dedupe exact bytes
    existing_sha = _dedupe_by_sha(db, sha256, content_length)
    if existing_sha:
        return {
            "status": "exists_sha",
            "image_id": existing_sha.id,
            "image_path": existing_sha.image_path,
            "sha256": existing_sha.sha256,
        }

    # Choose file extension from response headers
    ext = _ext_from_content_type(r.headers.get("Content-Type"))
    base_fn = _new_image_fn(db)
    filename = f"{base_fn}.{ext}"
    path = IMAGES_DIR / filename

    # Image conversion and standardization for saving
    img = Image.open(BytesIO(data)).convert('RGB')
    img = _resize_for_saving(img) 
    
    # Write image to disk
    _atomic_write(path, img)

    
    # Save DB row
    phash = _compute_phash(img)
    row = _save_image_row(
        db,
        user_id=user_id,
        sha256=sha256,
        phash=phash,
        image_url=str(payload.image_url),
        image_path=str(path),
        content_length=content_length,
    )

    return {
        "status": "inserted",
        "image_id": row.id,
        "image_path": row.image_path,
        "sha256": row.sha256,
    }


@router.post("/ingest_upload")
async def ingest_upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    user_id = current_user.id
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Not an image: {file.content_type}")
     
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    content_length = len(data)
    sha = _sha256_bytes(data)

    # exact dedupe
    existing = db.execute(
        select(models.Image).where(
            models.Image.sha256 == sha,
            models.Image.content_length == content_length,
        )
    ).scalar_one_or_none()
    if existing:
        return {"status": "exists_sha", "image_id": existing.id, "image_path": existing.image_path, "sha256": existing.sha256}

    
    # Get Filename for new 
    ext = _ext_from_content_type(file.content_type)
    base_fn = _new_image_fn(db)
    filename = f"{base_fn}.{ext}"
    path = IMAGES_DIR / filename

    # Image conversion and standardization for saving
    img = Image.open(BytesIO(data)).convert('RGB')
    img = _resize_for_saving(img) 

    # Write image to disk
    _atomic_write(path, img)

    
    # Save DB row
    phash = _compute_phash(img)

    row = models.Image(
        user_id=user_id,
        sha256=sha,
        phash=phash,   # placeholder for now
        image_url=None,            # uploads don’t have to have a URL
        image_path=str(path),
        content_length=content_length,
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
    except IntegrityError:
        db.rollback()
        existing = db.execute(select(models.Image).where(models.Image.sha256 == sha)).scalar_one()
        return {"status": "exists_sha", "image_id": existing.id, "image_path": existing.image_path, "sha256": existing.sha256}

    return {
        "status": "inserted",
        "image_id": row.id,
        "image_path": row.image_path,
        "sha256": row.sha256,
        "original_filename": file.filename,
    }



@router.get("/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(models.Image).where(models.Image.id == image_id)).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(row.image_path)


@router.get("/{image_id}/meta")
def get_image_meta(image_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(models.Image).where(models.Image.id == image_id)).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="image not found")
    return {"image_id": row.id, "image_path": row.image_path, "image_url": row.image_url}



class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    is_public: bool
    created_at: Optional[datetime] = None  # or datetime if you use it


@router.get("/me", response_model=list[ImageOut])
def list_my_images(
    public: Optional[bool] = Query(default=None, description="If set, filter by visibility"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    order: Literal["desc", "asc"] = Query(default="desc"),
    db: Session = Depends(get_db),                 # <- your DB session dependency
    user=Depends(get_current_user),               # <- your auth dependency
):
    q = db.query(models.Image).filter(models.Image.user_id == user.id)

    if public is not None:
        q = q.filter(models.Image.is_public == public)

    if order == "desc":
        q = q.order_by(models.Image.created_at.desc())
    else:
        q = q.order_by(models.Image.created_at.asc())

    images = q.offset(offset).limit(limit).all()
    return [ImageOut.model_validate(img) for img in images]


class ImageVisibilityUpdate(BaseModel):
    is_public: bool


@router.patch("/{image_id}", response_model=ImageOut)
def update_image_visibility(
    image_id: int,
    payload: ImageVisibilityUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    img = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Only the uploader can change visibility
    if img.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not allowed to modify this image")

    img.is_public = payload.is_public
    db.add(img)
    db.commit()
    db.refresh(img)

    return ImageOut.model_validate(img)



@router.get("/random_public", response_model=ImageOut)
def get_random_public_image(db: Session = Depends(get_db), user=Depends(get_current_user)):
    row = db.execute(
        select(models.Image)
        .where(models.Image.is_public == True)
        .order_by(func.random())
        .limit(1)
    ).scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="No public images found")

    return ImageOut.model_validate(row)

