import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
import httpx

from app.db import get_db
from app.models import Image, User
from app.deps import get_current_user
from pydantic import BaseModel

router = APIRouter(prefix="/api/questions", tags=["questions"])

MODEL_BASE = os.environ.get("MODEL_BASE", "http://127.0.0.1:9000").rstrip("/")
QUESTION_ROUTE = "chat/internvl2_5_2b"

QUESTION_PROMPT = """Generate a challenging visual question that requires precise observation.

Strong examples:
- "How many letters 'e' appear in the visible text?"
- "Which object is closest to the upper-right corner of the image?"
- "What color is the third item from the left on the top shelf?"
- "Count the number of people wearing glasses."
- "What text appears on the sign in the background?"
- "Which hand is the person using to hold the object?"
- "How many windows are visible on the building's second floor?"
- "What brand logo is partially visible on the item?"

Weak examples to avoid:
- "What is this image about?" (too vague)
- "Is there a person in the image?" (yes/no)
- "Describe what you see" (not a question)
- "What colors are present?" (too easy)

Generate ONE specific question about this image. The question should require counting, reading text, identifying spatial positions, or noticing small details. Output only the question."""


class QuestionResponse(BaseModel):
    image_id: int
    image_path: str
    question_text: str


class AnswerCreate(BaseModel):
    answer_text: str


@router.get("/next", response_model=QuestionResponse)
async def get_next_question(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a random image and generate a fresh question about it."""
    # Get random public image
    image = db.execute(
        select(Image).where(Image.is_public == True).order_by(func.random()).limit(1)
    ).scalar_one_or_none()
    
    # Generate question via model - always fresh
    try:
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{MODEL_BASE}/{QUESTION_ROUTE}",
                json={
                    "image_id": image.id,  # <-- was image_path, now image_id
                    "prompt": QUESTION_PROMPT,
                    "max_new_tokens": 256,
                    "do_sample": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            question_text = data.get("response", "").strip()
            
            # Fallback if model returns empty
            if not question_text:
                question_text = "How many distinct objects can you count in this image?"
                
    except Exception as e:
        print(f"Model error: {e}")
        question_text = "What is the most prominent text visible in this image?"
    
    return QuestionResponse(
        image_id=image.id,
        image_path=image.image_path,
        question_text=question_text,
    )


@router.post("/{image_id}/answer")
def submit_answer(
    image_id: int,
    data: AnswerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Store user's answer (for future training data)."""
    # Just log it for now - you can expand this later
    print(f"User {current_user.id} answered image {image_id}: {data.answer_text[:100]}")
    return {"status": "submitted"}