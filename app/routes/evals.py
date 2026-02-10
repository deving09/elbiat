from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.db import get_db
from app import models

from app.schemas import TaskCreate, ModelRegister, CreateEvalRun, LeaderboardEntry
from app.schemas import TaskResponse

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel




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
        task_dict = TaskResponse.from_orm(task).dict()
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
async def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    """Create a new evaluation task."""
    # Check for duplicate
    existing = db.execute(
        select(Task).where(Task.name == task.name)
    ).scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail=f"Task '{task.name}' already exists")
    
    db_task = Task(**task.dict())
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
        select(Model).order_by(Model.name)
    )
    return result.scalars().all()


@router.get("/models/{model_name}", response_model=ModelResponse)
async def get_model(model_name: str, db: Session = Depends(get_db)):
    """Get a specific model by name."""
    result = db.execute(
        select(Model).where(Model.name == model_name)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return model


@router.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(model: ModelCreate, db: Session = Depends(get_db)):
    """Create a new model entry."""
    existing = db.execute(
        select(Model).where(Model.name == model.name)
    ).scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail=f"Model '{model.name}' already exists")
    
    db_model = Model(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return db_model




# =============================================================================
# Eval Run Endpoints
# =============================================================================


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
        select(EvalRun, Model)
        .join(Model, EvalRun.model_id == Model.id)
        .where(EvalRun.task_id == task.id)
    )
    
    if status:
        query = query.where(EvalRun.status == status)
    
    query = query.order_by(desc(EvalRun.created_at)).offset(offset).limit(limit)
    
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
            "primary_metric": run.metrics.get(task.primary_metric_key) if run.metrics else None,
        }
        response.append(EvalRunResponse(**run_dict))
    
    return response



@router.get("/tasks/{task_name}/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    task_name: str,
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
        select(EvalRun, Model)
        .join(Model, EvalRun.model_id == Model.id)
        .where(EvalRun.task_id == task.id)
        .where(EvalRun.status == "completed")
        .where(EvalRun.metrics.isnot(None))
        .order_by(desc(EvalRun.created_at))
    )
    
    result = db.execute(query)
    rows = result.all()
    
    # Group by model, keep best result
    best_per_model = {}
    metric_key = task.primary_metric_key
    
    for run, model in rows:
        metric_val = run.metrics.get(metric_key) if run.metrics else None
        
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


@router.post("/tasks/{task_name}/runs", response_model=EvalRunResponse, status_code=201)
async def trigger_eval_run(
    task_name: str,
    run_request: EvalRunCreate,
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
        select(Model).where(Model.name == run_request.model_name)
    ).scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{run_request.model_name}' not found")
    
    # Create the run
    eval_run = EvalRun(
        task_id=task.id,
        model_id=model.id,
        status=EvalStatus.QUEUED,
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
        select(EvalRun, Task, Model)
        .join(Task, EvalRun.task_id == Task.id)
        .join(Model, EvalRun.model_id == Model.id)
        .where(EvalRun.id == run_id)
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
        primary_metric=run.metrics.get(task.primary_metric_key) if run.metrics else None
        #duration_seconds=(run.finished_at - run.started_at).total_seconds() if run.started_at and run.finished_at else None
    )