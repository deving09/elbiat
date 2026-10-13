"""
Side-by-side comparison routes.
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Inference stubs (connect to actual VLM later)
# ─────────────────────────────────────────────────────────────────────────────

def run_base_inference(image_id: int, prompt: str, base_model_id: int) -> str:
    """Run inference with the base model."""
    # TODO: Connect to actual VLM endpoint
    return f"[Base model response for image {image_id}]"


def run_adapted_inference(image_id: int, prompt: str, adaptation: models.UserAdaptation) -> str:
    """Run inference with user's adapted model."""
    # TODO: Connect to adapted VLM
    return f"[Adapted ({adaptation.level.value}) response for image {image_id}]"


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
    
    base_response = run_base_inference(request.image_id, request.prompt, adaptation.base_model_id)
    adapted_response = run_adapted_inference(request.image_id, request.prompt, adaptation)
    
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
    # For now, we'll need image_id and adaptation_id from the request
    # The schema has comparison_id but we should refactor to use image_id + adaptation_id
    
    feedback = models.ComparisonFeedback(
        user_id=current_user.id,
        image_id=feedback_in.comparison_id,  # TODO: refactor schema
        adaptation_id=1,  # TODO: get from request
        prompt="",  # TODO: get from request
        base_response="",  # TODO: store from comparison
        adapted_response="",  # TODO: store from comparison
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