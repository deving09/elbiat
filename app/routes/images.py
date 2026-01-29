import hashlib
import os
import random
import string
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from PIL import Image
from io import BytesIO
import numpy as np


from app import models
from app.db import get_db

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
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp)
    os.replace(tmp, path)




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
    user_id: int
    image_url: HttpUrl


# ----------------------------
# Endpoint 1: ingest from URL
# ----------------------------
@router.post("/ingest_url")
def ingest_url(payload: IngestUrlRequest, db: Session = Depends(get_db)):
    # Fast path: if exact same URL already exists, return it
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
    img = Image.Open(BytesIO(data)).convert('RGB')
    img = _resize_for_saving(img) 
    
    # Write image to disk
    _atomic_write(path, img)

    
    # Save DB row
    phash = _compute_phash(img)
    row = _save_image_row(
        db,
        user_id=payload.user_id,
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
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Not an image: {file.content_type}")
     
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    content_length = len(data)
    sha = sha256_bytes(data)

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
    img = Image.Open(BytesIO(data)).convert('RGB')
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
