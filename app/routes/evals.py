from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.db import get_db
from app import models

from app.schemas import TaskCreate, ModelRegister, CreateEvalRun, LeaderboardEntry
from app.schemas import TaskResponse, ModelResponse, EvalRunResponse

from app.models import Task, Evals, EvalStatus


from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

from app.deps import get_current_user

import numpy as np
import math



router = APIRouter(prefix="/api/evals", tags=["evaluations"])



# =============================================================================
# Task Endpoints
# =============================================================================

@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(db: Session = Depends(get_db)):
    """List all evaluation tasks."""
    result = db.execute(
        select(Task).order_by(Task.name)
    )
    tasks = result.scalars().all()
    
    # Add run counts
    response = []
    for task in tasks:
        #task_dict = TaskResponse.from_orm(task).dict()
        task_dict = TaskResponse.model_validate(task, from_attributes=True).model_dump()

        #count = db.execute(
        #    select(func.count(EvalRun.id)).where(EvalRun.task_id == task.id)
        #).scalar()
        #task_dict["run_count"] = count
        response.append(TaskResponse(**task_dict))
    
    return response


@router.get("/tasks/{task_name}", response_model=TaskResponse)
async def get_task(task_name: str, db: Session = Depends(get_db)):
    """Get a specific task by name."""
    result = db.execute(
        select(Task).where(Task.name == task_name)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    
    return task



@router.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(task: TaskCreate, 
    db: Session = Depends(get_db), 
    current_user=Depends(get_current_user)):
    """Create a new evaluation task."""
    # Check for duplicate
    existing = db.execute(
        select(Task).where(Task.name == task.name)
    ).scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail=f"Task '{task.name}' already exists")
     

    db_task = Task(user_id=current_user.id, **task.dict())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    return db_task




# =============================================================================
# Model Endpoints
# =============================================================================


@router.get("/models", response_model=List[ModelResponse])
async def list_models(db: Session = Depends(get_db)):
    """List all models."""
    result = db.execute(
        select(models.Models).order_by(models.Models.name)
    )
    return result.scalars().all()


@router.get("/models/{model_name}", response_model=ModelResponse)
async def get_model(model_name: str, db: Session = Depends(get_db)):
    """Get a specific model by name."""
    result = db.execute(
        select(models.Models).where(models.Model.name == model_name)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return model


@router.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(model: ModelRegister, db: Session = Depends(get_db)):
    """Create a new model entry."""
    existing = db.execute(
        select(models.Models).where(models.Models.name == model.name)
    ).scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail=f"Model '{model.name}' already exists")
    
    db_model = models.Models(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return db_model




# =============================================================================
# Eval Run Endpoints
# =============================================================================


"""
def get_metric_val(metrics, metric_key):

    metrics = sanitize_metrics(metrics)

    if metric_key == "avg":
        return np.mean(list(filter( lambda v: isinstance(v, float), metrics.values())))
    elif metric_key == "min":
        return np.min(list(filter( lambda v: isinstance(v, float), metrics.values())))
    elif metric_key == "max":
        return np.max(list(filter( lambda v: isinstance(v, float), metrics.values())))
    else:
        return metrics.get(metric_key) 
"""
def get_metric_val(metrics, metric_key):
    if not metrics:
        return None
    if metric_key == "avg":
        vals = [v for v in metrics.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        return np.mean(vals) if vals else None
    elif metric_key == "min":
        vals = [v for v in metrics.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        return np.min(vals) if vals else None
    elif metric_key == "max":
        vals = [v for v in metrics.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        return np.max(vals) if vals else None
    else:
        val = metrics.get(metric_key)
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

@router.get("/tasks/{task_name}/runs", response_model=List[EvalRunResponse])
async def list_task_runs(
    task_name: str,
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List evaluation runs for a task."""
    # Get task
    task = db.execute(
        select(Task).where(Task.name == task_name)
    ).scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    
    # Build query
    query = (
        select(Evals, models.Models)
        .join(models.Models, Evals.model_id == models.Models.id)
        .where(Evals.task_id == task.id)
    )
    
    if status:
        query = query.where(Evals.status == status)
    
    query = query.order_by(Evals.created_at.desc()).offset(offset).limit(limit)
    
    result = db.execute(query)
    rows = result.all()
    
    response = []
    for run, model in rows:
        run_dict = {
            "id": run.id,
            "task_id": run.task_id,
            "model_id": run.model_id,
            "status": run.status,
            "metrics": run.metrics,
            "artifacts_dir": run.artifacts_dir,
            "command": run.command,
            "git_commit": run.git_commit,
            "error": run.error,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "task_name": task.name,
            "model_name": model.name,
            "model_display_name": model.display_name,
            "primary_metric": get_metric_val(run.metrics, task.primary_metric_key)   
            #"primary_metric": run.metrics.get(task.primary_metric) if run.metrics else None,
        }
        response.append(EvalRunResponse(**run_dict))
    
    return response


def sanitize_metrics(metrics):
    """Replace NaN/Inf with None for JSON serialization."""
    if not metrics:
        return metrics
    
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (float,  np.floating)) and (math.isnan(value) or math.isinf(value)):
            sanitized[key] = None
        elif isinstance(value, dict):
            sanitized[key] = sanitize_metrics(value)
        else:
            sanitized[key] = value
    return sanitized


@router.get("/tasks/{task_name}/metrics", response_model=List[str])
async def get_available_metrics(
    task_name: str,
    db: Session = Depends(get_db)
):
    """
    Get all available metrics for a task by scanning completed runs.
    """
    task = db.execute(
        select(Task).where(Task.name == task_name)
    ).scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    
    # Get all unique metric keys from completed runs
    query = (
        select(Evals.metrics)
        .where(Evals.task_id == task.id)
        .where(Evals.status == "completed")
        .where(Evals.metrics.isnot(None))
    )
    
    result = db.execute(query)
    rows = result.scalars().all()
    
    # Collect all unique keys
    all_keys = set()
    for metrics in rows: 
        if metrics:
            metrics = sanitize_metrics(metrics)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_keys.add(key)

    # Add aggregate options + primary metric first
    ordered_metrics = ["avg", "min", "max"]
    if task.primary_metric_key:
        ordered_metrics.insert(0, task.primary_metric_key)
    
    # Add remaining keys (sorted)
    for key in sorted(all_keys):
        if key not in ordered_metrics:
            ordered_metrics.append(key)
    
    return ordered_metrics





@router.get("/tasks/{task_name}/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    task_name: str,
    metric: str = Query(None, description="Metric to sort by (default: task's primary metric). Use 'avg', 'min', 'max' for aggregates."),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """
    Get leaderboard for a task.
    Shows best result per model, sorted by primary metric.
    """
    task = db.execute(
        select(Task).where(Task.name == task_name)
    ).scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    
    # Get completed runs with metrics
    query = (
        select(Evals, models.Models)
        .join(models.Models, Evals.model_id == models.Models.id)
        .where(Evals.task_id == task.id)
        .where(Evals.status == "completed")
        .where(Evals.metrics.isnot(None))
        .order_by(Evals.created_at.desc())
        #.order_by(desc(Evals.created_at))
    )
    
    result = db.execute(query)
    rows = result.all()
    
    # Group by model, keep best result
    best_per_model = {}

    metric_key = metric if metric else task.primary_metric_key


    #metric_key = task.primary_metric_key
    
    for run, model in rows:
        #metric_val = run.metrics.get(metric_key) if run.metrics else None
        metrics = sanitize_metrics(run.metrics)
        metric_val = get_metric_val(metrics, metric_key) 
        if model.name not in best_per_model:
            best_per_model[model.name] = (run, model, metric_val)
        else:
            existing_val = best_per_model[model.name][2]
            if metric_val is not None:
                if existing_val is None or metric_val > existing_val:
                    best_per_model[model.name] = (run, model, metric_val)
    
    # Sort by metric (descending) and build response
    sorted_entries = sorted(
        best_per_model.values(),
        key=lambda x: x[2] if x[2] is not None else -999999,
        reverse=True
    )
    
    leaderboard = []
    for run, model, metric_val in sorted_entries[:limit]:
        #print(metric_val)
        leaderboard.append(LeaderboardEntry(
            model_name=model.name,
            model_display_name=model.display_name,
            primary_metric=metric_val,
            run_id=run.id,
            run_date=run.created_at,
            git_commit=run.git_commit,
            status=run.status
        ))
    
    return leaderboard


class TriggerEvalRequest(BaseModel):
    model_name: str


@router.post("/tasks/{task_name}/runs", response_model=EvalRunResponse, status_code=201)
async def trigger_eval_run(
    task_name: str,
    run_request: TriggerEvalRequest,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_user)  # Add auth
):
    """
    Trigger a new evaluation run.
    Creates a queued run that the worker will pick up.
    """
    # Get task
    task = db.execute(
        select(Task).where(Task.name == task_name)
    ).scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    
    # Get model
    model = db.execute(
        select(models.Models).where(models.Models.name == run_request.model_name)
    ).scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{run_request.model_name}' not found")
    
    # Create the run
    eval_run = Evals(
        task_id=task.id,
        model_id=model.id,
        status=EvalStatus.QUEUED,
        metrics={},
        #config_snapshot=run_request.config_overrides,
        # created_by_user_id=current_user.id  # Add when you have auth
    )
    
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    
    return EvalRunResponse(
        id=eval_run.id,
        task_id=eval_run.task_id,
        model_id=eval_run.model_id,
        status=eval_run.status,
        created_at=eval_run.created_at,
        task_name=task.name,
        model_name=model.name,
        model_display_name=model.display_name
    )


@router.get("/runs/{run_id}", response_model=EvalRunResponse)
async def get_run(run_id: int, db: Session = Depends(get_db)):
    """Get details of a specific run."""
    result = db.execute(
        select(Evals, Task, models.Models)
        .join(Task, Evals.task_id == Task.id)
        .join(models.Models, Evals.model_id == Model.id)
        .where(Evals.id == run_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    run, task, model = row
    
    return EvalRunResponse(
        id=run.id,
        task_id=run.task_id,
        model_id=run.model_id,
        status=run.status,
        metrics=run.metrics,
        artifacts_dir=run.artifacts_dir,
        command=run.command,
        git_commit=run.git_commit,
        error=run.error,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        task_name=task.name,
        model_name=model.name,
        model_display_name=model.display_name,
        primary_metric=get_metric_val(run.metrics, task.primary_metric_key)
        #primary_metric=run.metrics.get(task.primary_metric) if run.metrics else None
        #duration_seconds=(run.finished_at - run.started_at).total_seconds() if run.started_at and run.finished_at else None
    )