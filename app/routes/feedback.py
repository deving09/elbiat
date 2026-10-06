from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc, asc, func
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

from app.db import get_db
from app.deps import get_current_user
from app import models

router = APIRouter(prefix="/feedback", tags=["feedback"])



# Add new schema for attribution
class AttributionScore(BaseModel):
    benchmark: str
    method: str
    checkpoint_id: str
    influence_score: float
    z_score: Optional[float]
    rank: Optional[int]


class ConvoOut(BaseModel):
    id: int
    image_id: int
    conversations: list[dict]
    model_name: str
    model_type: str
    task: str
    feedback: str
    enabled: bool
    monetized: bool
    created_at: datetime
    
    # Computed fields
    prompt: Optional[str] = None
    response: Optional[str] = None
    feedback_length: int = 0
    attribution_score: float = 0.0  # Placeholder, will come from separate table later

    attributions: list[AttributionScore] = []  # NEW: list of attribution scores
    
    class Config:
        from_attributes = True


class ConvoUpdate(BaseModel):
    feedback: Optional[str] = None
    enabled: Optional[bool] = None


class FeedbackListResponse(BaseModel):
    items: list[ConvoOut]
    total: int
    page: int
    page_size: int


def extract_prompt_response(conversations: list[dict]) -> tuple[str, str]:
    """Extract prompt and response from conversations JSON."""
    prompt = ""
    response = ""
    for turn in conversations:
        if turn.get("from") == "human":
            prompt = turn.get("value", "").replace("<image>\n", "").strip()
        elif turn.get("from") == "gpt":
            response = turn.get("value", "").strip()
    return prompt, response



@router.get("/attribution-filters")
def get_attribution_filters(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Get available benchmarks, checkpoints, and methods for filtering."""
    # Get distinct values from data_attributions for user's convos
    user_convo_ids = select(models.Convo.id).where(models.Convo.user_id == current_user.id)
    
    benchmarks = db.execute(
        select(models.DataAttribution.benchmark.distinct())
        .where(models.DataAttribution.convo_id.in_(user_convo_ids))
    ).scalars().all()
    
    checkpoints = db.execute(
        select(models.DataAttribution.checkpoint_id.distinct())
        .where(models.DataAttribution.convo_id.in_(user_convo_ids))
    ).scalars().all()
    
    methods = db.execute(
        select(models.DataAttribution.method.distinct())
        .where(models.DataAttribution.convo_id.in_(user_convo_ids))
    ).scalars().all()
    
    return {
        "benchmarks": benchmarks,
        "checkpoints": checkpoints,
        "methods": methods,
    }


@router.get("", response_model=FeedbackListResponse)
def list_feedback(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort_by: Literal["created_at", "feedback_length", "attribution_score"] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    enabled_only: bool = Query(default=False),
    # NEW: attribution filters
    benchmark: Optional[str] = Query(default=None),
    checkpoint_id: Optional[str] = Query(default=None),
    method: str = Query(default="tracin"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """List user's feedback with sorting and pagination."""
    
    query = select(models.Convo).where(models.Convo.user_id == current_user.id)
    
    if enabled_only:
        query = query.where(models.Convo.enabled == True)
    
    order_func = desc if sort_order == "desc" else asc
    
    if sort_by == "attribution_score" and benchmark and checkpoint_id:
        query = query.outerjoin(
            models.DataAttribution,
            (models.Convo.id == models.DataAttribution.convo_id) &
            (models.DataAttribution.benchmark == benchmark) &
            (models.DataAttribution.checkpoint_id == checkpoint_id) &
            (models.DataAttribution.method == method)
        )
        order_col = models.DataAttribution.influence_score
        if sort_order == "desc":
            query = query.order_by(order_col.desc().nullslast())
        else:
            query = query.order_by(order_col.asc().nullsfirst())
    else:
        order_func = desc if sort_order == "desc" else asc
        if sort_by == "created_at":
            query = query.order_by(order_func(models.Convo.created_at))
        elif sort_by == "feedback_length":
            query = query.order_by(order_func(func.length(models.Convo.feedback)))

    
    
    # Total count
    count_query = select(func.count()).select_from(models.Convo).where(
        models.Convo.user_id == current_user.id
    )
    if enabled_only:
        count_query = count_query.where(models.Convo.enabled == True)
    total = db.execute(count_query).scalar()
    
    # Paginate
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    convos = db.execute(query).scalars().all()
    
    items = []
    for convo in convos:

        # Query attributions for this convo
        attr_query = select(models.DataAttribution).where(
            models.DataAttribution.convo_id == convo.id
        )
        if benchmark:
            attr_query = attr_query.where(models.DataAttribution.benchmark == benchmark)
        if checkpoint_id:
            attr_query = attr_query.where(models.DataAttribution.checkpoint_id == checkpoint_id)
        if method:
            attr_query = attr_query.where(models.DataAttribution.method == method)
        
        attrs = db.execute(attr_query).scalars().all()


        prompt, response = extract_prompt_response(convo.conversations)
        item = ConvoOut(
            id=convo.id,
            image_id=convo.image_id,
            conversations=convo.conversations,
            model_name=convo.model_name,
            model_type=convo.model_type,
            task=convo.task,
            feedback=convo.feedback,
            enabled=convo.enabled,
            monetized=convo.monetized,
            created_at=convo.created_at,
            prompt=prompt,
            response=response,
            feedback_length=len(convo.feedback) if convo.feedback else 0,
            attribution_score=0.0,  # Placeholder
            attributions=[
                AttributionScore(
                    benchmark=a.benchmark,
                    method=a.method,
                    checkpoint_id=a.checkpoint_id,
                    influence_score=a.influence_score,
                    z_score=a.z_score,
                    rank=a.rank,
                )
                for a in attrs
            ],
        )
        items.append(item)
    
    return FeedbackListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{convo_id}", response_model=ConvoOut)
def get_feedback(
    convo_id: int,
    benchmark: Optional[str] = Query(default=None),
    checkpoint_id: Optional[str] = Query(default=None),
    method: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Get a single feedback item."""
    convo = db.execute(
        select(models.Convo).where(
            models.Convo.id == convo_id,
            models.Convo.user_id == current_user.id,
        )
    ).scalar_one_or_none()
    
    if not convo:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Query attributions for this convo
    attr_query = select(models.DataAttribution).where(
        models.DataAttribution.convo_id == convo.id
    )
    if benchmark:
        attr_query = attr_query.where(models.DataAttribution.benchmark == benchmark)
    if checkpoint_id:
        attr_query = attr_query.where(models.DataAttribution.checkpoint_id == checkpoint_id)
    if method:
        attr_query = attr_query.where(models.DataAttribution.method == method)
    
    attrs = db.execute(attr_query).scalars().all()
    
    prompt, response = extract_prompt_response(convo.conversations)
    
    return ConvoOut(
        id=convo.id,
        image_id=convo.image_id,
        conversations=convo.conversations,
        model_name=convo.model_name,
        model_type=convo.model_type,
        task=convo.task,
        feedback=convo.feedback,
        enabled=convo.enabled,
        monetized=convo.monetized,
        created_at=convo.created_at,
        prompt=prompt,
        response=response,
        feedback_length=len(convo.feedback) if convo.feedback else 0,
        attribution_score=0.0,
        attributions=[
            AttributionScore(
                benchmark=a.benchmark,
                method=a.method,
                checkpoint_id=a.checkpoint_id,
                influence_score=a.influence_score,
                z_score=a.z_score,
                rank=a.rank,
            )
            for a in attrs
        ],
    )


@router.patch("/{convo_id}", response_model=ConvoOut)
def update_feedback(
    convo_id: int,
    update: ConvoUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Update feedback text or enabled status."""
    convo = db.execute(
        select(models.Convo).where(
            models.Convo.id == convo_id,
            models.Convo.user_id == current_user.id,
        )
    ).scalar_one_or_none()
    
    if not convo:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    if update.feedback is not None:
        convo.feedback = update.feedback
    if update.enabled is not None:
        convo.enabled = update.enabled
    
    db.commit()
    db.refresh(convo)
    
    prompt, response = extract_prompt_response(convo.conversations)
    
    return ConvoOut(
        id=convo.id,
        image_id=convo.image_id,
        conversations=convo.conversations,
        model_name=convo.model_name,
        model_type=convo.model_type,
        task=convo.task,
        feedback=convo.feedback,
        enabled=convo.enabled,
        monetized=convo.monetized,
        created_at=convo.created_at,
        prompt=prompt,
        response=response,
        feedback_length=len(convo.feedback) if convo.feedback else 0,
        attribution_score=0.0,
    )


@router.delete("/{convo_id}")
def delete_feedback(
    convo_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Delete a feedback item."""
    convo = db.execute(
        select(models.Convo).where(
            models.Convo.id == convo_id,
            models.Convo.user_id == current_user.id,
        )
    ).scalar_one_or_none()
    
    if not convo:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    db.delete(convo)
    db.commit()
    
    return {"status": "deleted", "id": convo_id}

