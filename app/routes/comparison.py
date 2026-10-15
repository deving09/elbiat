"""
Side-by-side comparison routes.
"""

import os
import httpx

import logging
import time

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app import models
from app.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    ComparisonFeedbackCreate,
    ComparisonFeedbackResponse,
)
from app.deps import get_current_user

router = APIRouter(prefix="/api/comparison", tags=["comparison"])

logger = logging.getLogger("uvicorn.error")

# ─────────────────────────────────────────────────────────────────────────────
# Inference via model service (port 9000)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:9000")
MODEL_SERVICE_TIMEOUT = float(os.getenv("MODEL_SERVICE_TIMEOUT", "120"))

# Fallback when an adaptation row has no config["model"]
LEVEL_MODEL_MAP = {
    "lora": "dpo-v1",
    "full": "feedback-lora",
    "rag": "base",
}


def call_model_service(model_name: str, image_id: int, prompt: str) -> str:
    try:
        resp = httpx.post(
            f"{MODEL_SERVICE_URL}/chat",
            params={"model": model_name},
            json={
                "image_id": image_id,
                "prompt": prompt,
                "history": None,
                "max_new_tokens": 1024,
                "do_sample": False,
            },
            timeout=MODEL_SERVICE_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Model service error ({model_name}): {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Model service unavailable ({model_name}): {e}")
    return resp.json().get("response", "")


def run_base_inference(image_id: int, prompt: str, base_model_id: int) -> str:
    """Run inference with the base model."""
    return call_model_service("base", image_id, prompt)


def run_adapted_inference(image_id: int, prompt: str, adaptation: models.UserAdaptation) -> str:
    """Run inference with the adaptation's model (config.model, else by level)."""
    model_name = (adaptation.config or {}).get("model") \
        or LEVEL_MODEL_MAP.get(adaptation.level.value, "base")
    return call_model_service(model_name, image_id, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=ComparisonResponse)
def run_comparison(
    request: ComparisonRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Run side-by-side inference: base model vs adapted model."""
    if request.adaptation_id:
        adaptation = db.get(models.UserAdaptation, request.adaptation_id)
        if not adaptation:
            raise HTTPException(status_code=404, detail="Adaptation not found")
        if adaptation.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        if adaptation.status != models.AdaptationStatus.READY:
            raise HTTPException(status_code=400, detail=f"Adaptation not ready (status={adaptation.status.value})")
    else:
        adaptation = db.query(models.UserAdaptation).filter(
            models.UserAdaptation.user_id == current_user.id,
            models.UserAdaptation.status == models.AdaptationStatus.READY,
        ).order_by(models.UserAdaptation.created_at.desc()).first()
        
        if not adaptation:
            raise HTTPException(status_code=404, detail="No ready adaptations found")
    
    #base_response = run_base_inference(request.image_id, request.prompt, adaptation.base_model_id)
    #adapted_response = run_adapted_inference(request.image_id, request.prompt, adaptation)
    t0 = time.perf_counter()
    base_response = run_base_inference(request.image_id, request.prompt, adaptation.base_model_id)
    t1 = time.perf_counter()
    logger.info("[compare] base: %.1fs, %d chars: %r",
                t1 - t0, len(base_response), base_response[:200])

    adapted_response = run_adapted_inference(request.image_id, request.prompt, adaptation)
    t2 = time.perf_counter()
    logger.info("[compare] adapted (id=%s level=%s config=%s): %.1fs, %d chars: %r",
                adaptation.id, adaptation.level.value, adaptation.config,
                t2 - t1, len(adapted_response), adapted_response[:200])
    logger.info("[compare] total: %.1fs", t2 - t0)

    
    return ComparisonResponse(
        image_id=request.image_id,
        prompt=request.prompt,
        base_response=base_response,
        adapted_response=adapted_response,
        adaptation_id=adaptation.id,
        adaptation_level=adaptation.level.value,
    )



@router.post("/feedback", response_model=ComparisonFeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_comparison_feedback(
    feedback_in: ComparisonFeedbackCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Submit feedback on a comparison."""
    # Verify adaptation belongs to user
    adaptation = db.get(models.UserAdaptation, feedback_in.adaptation_id)
    if not adaptation:
        raise HTTPException(status_code=404, detail="Adaptation not found")
    if adaptation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    feedback = models.ComparisonFeedback(
        user_id=current_user.id,
        image_id=feedback_in.image_id,
        adaptation_id=feedback_in.adaptation_id,
        prompt=feedback_in.prompt,
        base_response=feedback_in.base_response,
        adapted_response=feedback_in.adapted_response,
        preference=feedback_in.preference,
        base_thumbs=feedback_in.base_thumbs,
        adapted_thumbs=feedback_in.adapted_thumbs,
        base_edit=feedback_in.base_edit,
        adapted_edit=feedback_in.adapted_edit,
        comment=feedback_in.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    
    return feedback