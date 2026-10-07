from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app import models, schemas


router = APIRouter(prefix="/api/chat-feedback", tags=["chat-feedback"])


def get_feedback_config(experiment_bucket: Optional[str]) -> List[str]:
    """
    Return the list of available feedback actions based on experiment bucket.

    Bucket configurations:
    - "control": thumbs only
    - "edit": thumbs + edit
    - "comment": thumbs + comment
    - "full": thumbs + edit + comment
    - None/default: thumbs only
    """
    configs = {
        "control": ["thumbs"],
        "edit": ["thumbs", "edit"],
        "comment": ["thumbs", "comment"],
        "full": ["thumbs", "edit", "comment"],
    }
    return configs.get(experiment_bucket, ["thumbs"])


# ─────────────────────────────────────────────────────────────────────────────
# Feedback endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/submit", response_model=schemas.FeedbackSubmitResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    data: schemas.FeedbackSubmit,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Submit feedback for a chat. Creates a Convo if this is the first feedback.
    """
    available_actions = get_feedback_config(current_user.experiment_bucket)
    
    # Validate feedback types against user's bucket
    if data.thumbs and "thumbs" not in available_actions:
        raise HTTPException(status_code=403, detail="Thumbs feedback not available")
    if (data.edit_original or data.edit_revised) and "edit" not in available_actions:
        raise HTTPException(status_code=403, detail="Edit feedback not available")
    if data.comment_text and "comment" not in available_actions:
        raise HTTPException(status_code=403, detail="Comments not available")
    
    # Get or create chat session
    if data.chat_session_id:
        # Existing session - verify ownership
        chat_session = db.get(models.ChatSession, data.chat_session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        if chat_session.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
    else:
        # Create new chat session from messages
        if not data.messages:
            raise HTTPException(status_code=400, detail="Messages required when creating new chat session")

        # Generate title from first user message if not provided
        title = data.title
        if not title:
            for msg in data.messages:
                if msg.role == "user":
                    title = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
                    break

        chat_session = models.ChatSession(
            user_id=current_user.id,
            title=title,
            messages=[{"role": m.role, "content": m.content} for m in data.messages],
        )
        db.add(chat_session)
        db.flush()
    
    # Create feedback record
    feedback = models.ConvoFeedback(
        chat_session_id=chat_session.id,
        user_id=current_user.id,
        turn_index=data.turn_index,
        thumbs=data.thumbs,
        edit_original=data.edit_original,
        edit_revised=data.edit_revised,
        experiment_bucket=current_user.experiment_bucket,
        available_actions=available_actions,
    )
    db.add(feedback)
    db.flush()
    
    # Optionally create comment
    comment_id = None
    if data.comment_text:
        comment = models.ConvoComment(
            chat_session_id=chat_session.id, 
            user_id=current_user.id,
            turn_index=data.turn_index,
            text=data.comment_text,
            experiment_bucket=current_user.experiment_bucket,
        )
        db.add(comment)
        db.flush()
        comment_id = comment.id
    
    db.commit()
    
    return schemas.FeedbackSubmitResponse(
        chat_session_id=chat_session.id, 
        feedback_id=feedback.id,
        comment_id=comment_id,
    )



@router.post("/", response_model=schemas.ChatFeedbackResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(
    feedback_in: schemas.ChatFeedbackCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Submit feedback for a conversation turn."""
    # Verify convo exists and belongs to user
    convo = db.get(models.Convo, feedback_in.convo_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if convo.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to provide feedback on this conversation")

    # Get available actions for this user's bucket
    available_actions = get_feedback_config(current_user.experiment_bucket)

    # Validate that the submitted feedback type is allowed
    if feedback_in.thumbs and "thumbs" not in available_actions:
        raise HTTPException(status_code=403, detail="Thumbs feedback not available for your account")
    if (feedback_in.edit_original or feedback_in.edit_revised) and "edit" not in available_actions:
        raise HTTPException(status_code=403, detail="Edit feedback not available for your account")

    feedback = models.ConvoFeedback(
        convo_id=feedback_in.convo_id,
        user_id=current_user.id,
        turn_index=feedback_in.turn_index,
        thumbs=feedback_in.thumbs,
        edit_original=feedback_in.edit_original,
        edit_revised=feedback_in.edit_revised,
        experiment_bucket=current_user.experiment_bucket,
        available_actions=available_actions,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


@router.get("/convo/{convo_id}", response_model=List[schemas.ChatFeedbackResponse])
def get_convo_feedback(
    convo_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Get all feedback for a conversation."""
    convo = db.get(models.Convo, convo_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if convo.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return db.query(models.ConvoFeedback).filter(
        models.ConvoFeedback.convo_id == convo_id
    ).order_by(models.ConvoFeedback.turn_index).all()


@router.get("/config")
def get_feedback_config_endpoint(
    current_user: models.User = Depends(get_current_user),
):
    """Get the feedback configuration for the current user."""
    actions = get_feedback_config(current_user.experiment_bucket)
    return {
        "experiment_bucket": current_user.experiment_bucket,
        "available_actions": actions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comment endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/comments", response_model=schemas.ChatCommentResponse, status_code=status.HTTP_201_CREATED)
def create_comment(
    comment_in: schemas.ChatCommentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Submit a comment on a conversation turn."""
    # Check if comments are enabled for this user
    available_actions = get_feedback_config(current_user.experiment_bucket)
    if "comment" not in available_actions:
        raise HTTPException(status_code=403, detail="Comments not available for your account")

    # Verify convo exists and belongs to user
    convo = db.get(models.Convo, comment_in.convo_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if convo.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    comment = models.ConvoComment(
        convo_id=comment_in.convo_id,
        user_id=current_user.id,
        turn_index=comment_in.turn_index,
        text=comment_in.text,
        experiment_bucket=current_user.experiment_bucket,
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


@router.get("/comments/convo/{convo_id}", response_model=List[schemas.ChatCommentResponse])
def get_convo_comments(
    convo_id: int,
    turn_index: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Get comments for a conversation, optionally filtered by turn."""
    convo = db.get(models.Convo, convo_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if convo.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    query = db.query(models.ConvoComment).filter(
        models.ConvoComment.convo_id == convo_id
    )
    if turn_index is not None:
        query = query.filter(models.ConvoComment.turn_index == turn_index)

    return query.order_by(models.ConvoComment.created_at).all()