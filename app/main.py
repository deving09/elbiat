from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from .db import SessionLocal
from . import models
from sqlalchemy import select, func
from .schemas import ConvoCreate, ImageCreate, ImgHashCheck
from sqlalchemy.exc import IntegrityError

import random
import string

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/users")
def create_user(email: str, db: Session = Depends(get_db)):

    existing = db.execute(
            select(models.User).where(models.User.email == email)
            ).scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")

    user = models.User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email}


@app.post("/convos")
def add_convo(
        convo_in: ConvoCreate,
        db: Session = Depends(get_db)
    ):
    convo = models.Convo(**convo_in.model_dump())
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return {"id": convo.id, "feedback":convo.feedback}


@app.get("/img_url_check")
def check_img_url(image_url: str, db: Session = Depends(get_db)):

    img = db.execute(
            select(models.Image).where(models.Image.image_url == image_url)
            ).scalar_one_or_none()

    if img:
        return {"found": True, "image_id":img.id, "filename": img.image_path}

    return {"found": False, "image_id": None, "filename": None}


@app.post("/img_new_fn")
def new_image_fn(images_folder: str="images", db: Session = Depends(get_db)):

    max_id = db.execute(
        select(func.max(models.Image.id))
    ).scalar()

    next_id = (max_id or 0) + 1

    suffix = "".join(random.choices(string.ascii_lowercase, k=4))

    filename = f"{images_folder}/{next_id}_{suffix}"
    return {"filename": filename}


@app.post("/save_img_info")
def save_img_info(
    data: ImageCreate,
    db: Session = Depends(get_db),
):
    existing = db.execute(
        select(models.Image).where(models.Image.sha256 == data.sha256)
    ).scalar_one_or_none()

    if existing:
        return {
            "status": "exists",
            "image_hash_id": existing.id,
        }

    row = models.Image(
        sha256=data.sha256,
        phash=data.phash,
        image_url=data.image_url,
        image_path=data.image_path,
        content_length=data.content_length,
    )

    db.add(row)
    try:
        db.commit()
        db.refresh(row)
    except IntegrityError:
        db.rollback()
        # race condition fallback
        existing = db.execute(
            select(models.Image).where(models.Image.sha256 == data.sha256)
        ).scalar_one()
        return {
            "status": "exists",
            "image_id": existing.id,
        }

    return {
        "status": "inserted",
        "image_id": row.id,
    }



@app.post("/img_hash_check")
def img_hash_check(
    data: ImgHashCheck,
    check_type: str = Query(default="sha256", pattern="^(sha256|phash)$"),
    db: Session = Depends(get_db),
):
    if check_type == "sha256":
        if not data.sha256:
            raise HTTPException(status_code=400, detail="sha256 is required for sha256 check")
        existing = db.execute(
            select(
                models.Image.id).where(models.Image.sha256 == data.sha256,
                models.Image.content_length == data.content_length)
        ).first()
        return {"found": existing is not None}

    # placeholder for later
    # you can implement: fetch candidate phashes and compute hamming distance
    if not data.phash:
        raise HTTPException(status_code=400, detail="phash is required for phash check")
    return {"found": False, "note": "phash check not implemented yet"}





