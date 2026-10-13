"""
User adaptation routes for personalized model variants.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app import models
from app.schemas import TriggerAdaptationRequest, UserAdaptationResponse
from app.deps import get_current_user

router = APIRouter(prefix="/api/adaptations", tags=["adaptations"])


# ─────────────────────────────────────────────────────────────────────────────
# Background task stubs
# ─────────────────────────────────────────────────────────────────────────────

def train_soft_prompt(adaptation_id: int):
    """Background task: train soft prompt from user's interaction history."""
    # TODO: Implement
    pass


def train_lora_adapter(adaptation_id: int):
    """Background task: train LoRA adapter from user's contribution data."""
    # TODO: Implement
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/trigger", response_model=UserAdaptationResponse, status_code=status.HTTP_201_CREATED)
def trigger_adaptation(
    request: TriggerAdaptationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Trigger creation of a user-specific adaptation."""
    existing = db.query(models.UserAdaptation).filter(
        models.UserAdaptation.user_id == current_user.id,
        models.UserAdaptation.base_model_id == request.base_model_id,
        models.UserAdaptation.level == request.level,
        models.UserAdaptation.status.in_([
            models.AdaptationStatus.PENDING,
            models.AdaptationStatus.TRAINING,
            models.AdaptationStatus.READY
        ])
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"You already have a {request.level.value} adaptation for this model (id={existing.id})"
        )
    
    # Count user's image contributions
    contribution_count = db.query(func.count(models.Image.id)).filter(
        models.Image.user_id == current_user.id
    ).scalar() or 0
    
    # RAG is instantly ready
    initial_status = (
        models.AdaptationStatus.READY 
        if request.level == models.AdaptationLevel.RAG 
        else models.AdaptationStatus.PENDING
    )
    
    adaptation = models.UserAdaptation(
        user_id=current_user.id,
        base_model_id=request.base_model_id,
        level=request.level,
        status=initial_status,
        config=request.config,
        contribution_count=contribution_count,
    )
    db.add(adaptation)
    db.commit()
    db.refresh(adaptation)
    
    # Queue background training for non-RAG
    if request.level == models.AdaptationLevel.SOFT_PROMPT:
        adaptation.status = models.AdaptationStatus.TRAINING
        db.commit()
        background_tasks.add_task(train_soft_prompt, adaptation.id)
    elif request.level == models.AdaptationLevel.LORA:
        adaptation.status = models.AdaptationStatus.TRAINING
        db.commit()
        background_tasks.add_task(train_lora_adapter, adaptation.id)
    
    return adaptation


@router.get("/", response_model=List[UserAdaptationResponse])
def list_adaptations(
    level: Optional[models.AdaptationLevel] = None,
    status: Optional[models.AdaptationStatus] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """List all adaptations for the current user."""
    query = db.query(models.UserAdaptation).filter(
        models.UserAdaptation.user_id == current_user.id
    )
    
    if level:
        query = query.filter(models.UserAdaptation.level == level)
    if status:
        query = query.filter(models.UserAdaptation.status == status)
    
    return query.order_by(models.UserAdaptation.created_at.desc()).all()


@router.get("/{adaptation_id}", response_model=UserAdaptationResponse)
def get_adaptation(
    adaptation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Get details of a specific adaptation."""
    adaptation = db.get(models.UserAdaptation, adaptation_id)
    
    if not adaptation:
        raise HTTPException(status_code=404, detail="Adaptation not found")
    if adaptation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return adaptation


@router.delete("/{adaptation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_adaptation(
    adaptation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Delete an adaptation."""
    adaptation = db.get(models.UserAdaptation, adaptation_id)
    
    if not adaptation:
        raise HTTPException(status_code=404, detail="Adaptation not found")
    if adaptation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    db.delete(adaptation)
    db.commit()
    return None