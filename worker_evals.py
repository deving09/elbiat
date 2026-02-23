#!/usr/bin/env python3
"""
Worker process for running VLMEvalKit evaluations.

This worker:
1. Polls the database for queued eval runs
2. Executes VLMEvalKit with the appropriate parameters
3. Parses output metrics and updates the database
4. Handles errors gracefully

Run with: python worker_evals.py
Or as a systemd service for production.
"""

import os
import sys
import json
import time
import glob
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import csv

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text

from app.models import EvalStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('worker_evals.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration - adjust these to your environment
# =============================================================================

# Database URL (use same as your FastAPI app)
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://postgres:postgres@localhost:5432/app"
)

# VLMEvalKit paths
VLMEVAL_ROOT = os.getenv(
    "VLMEVAL_ROOT",
    "/home/ubuntu/workspace/elbiat/external/VLMEvalKit"
)
VLMEVAL_OUTPUTS = os.path.join(VLMEVAL_ROOT, "outputs")
VLMEVAL_RUN_SCRIPT = os.path.join(VLMEVAL_ROOT, "run.py")

# Worker settings
POLL_INTERVAL_SECONDS = 5
MAX_RETRIES = 3

# =============================================================================
# Database Models (import from your app or define here)
# =============================================================================

# If your models are in app/models.py, you can import them:
# from app.models import EvalRun, Task, Model, EvalStatus

# For standalone operation, we'll use raw SQL or define minimal models here
from enum import Enum as PyEnum


"""
class EvalStatus(str, PyEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "completed"
    FAILED = "failed"
"""

# =============================================================================
# Database Operations
# =============================================================================

class EvalWorkerDB:
    """Database operations for the eval worker."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        return self.SessionLocal()
    
    def get_next_queued_run(self) -> Optional[dict]:
        """Get the oldest queued eval run with task and model info."""
        with self.get_session() as session:
            result = session.execute(
                text("""
                SELECT 
                    er.id,
                    er.task_id,
                    er.model_id,
                    t.name as task_name,
                    t.vlmeval_data,
                    t.primary_metric_key,
                    t.primary_metric_suffix,
                    m.name as model_name,
                    m.vlmeval_model
                FROM eval_runs er
                JOIN tasks t ON er.task_id = t.id
                JOIN models m ON er.model_id = m.id
                WHERE er.status = :status
                ORDER BY er.created_at ASC
                LIMIT 1
                """),
                {"status": "QUEUED"} #EvalStatus.QUEUED}
            )
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    
    def mark_running(self, run_id: int, artifacts_dir: str, command: str, git_commit: str = None):
        """Mark a run as running."""
        with self.get_session() as session:
            session.execute(
                text("""
                UPDATE eval_runs 
                SET status = :status,
                    started_at = :started_at,
                    artifacts_dir = :artifacts_dir,
                    command = :command,
                    git_commit = :git_commit
                WHERE id = :id
                """),
                {
                    "status": "RUNNING",
                    "started_at": datetime.utcnow(),
                    "artifacts_dir": artifacts_dir,
                    "command": command,
                    "git_commit": git_commit,
                    "id": run_id
                }
            )
            session.commit()
    
    def mark_completed(self, run_id: int, metrics: dict):
        """Mark a run as completed with metrics."""
        with self.get_session() as session:
            session.execute(
                text("""
                UPDATE eval_runs 
                SET status = :status,
                    finished_at = :finished_at,
                    metrics = :metrics
                WHERE id = :id
                """),
                {
                    "status": EvalStatus.COMPLETE.name,
                    #"status": "completed", # EvalStatus.COMPLETED.name,
                    "finished_at": datetime.utcnow(),
                    "metrics": json.dumps(metrics),
                    "id": run_id
                }
            )
            session.commit()
    
    def mark_failed(self, run_id: int, error: str):
        """Mark a run as failed with error message."""
        with self.get_session() as session:
            session.execute(
                text("""
                UPDATE eval_runs 
                SET status = :status,
                    finished_at = :finished_at,
                    error = :error
                WHERE id = :id
                """),
                {
                    "status": "FAILED",
                    "finished_at": datetime.utcnow(),
                    "error": error[:4000] if error else None,  # Truncate if too long
                    "id": run_id
                }
            )
            session.commit()


# =============================================================================
# VLMEvalKit Integration
# =============================================================================

def get_git_commit(repo_path: str) -> Optional[str]:
    """Get the current git commit hash of the VLMEvalKit repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
        return None


def find_latest_run_dir(model_output_dir: str) -> Optional[str]:
    """Find the most recent T*_G* run directory for a model."""
    pattern = os.path.join(model_output_dir, "T*_G*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    # Sort by modification time, newest first
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def parse_acc_csv(csv_path: str) -> dict:
    """Parse a VLMEvalKit *_acc.csv file into metrics dict."""
    metrics = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                # Usually just one row with all metrics
                for row in rows:
                    for key, value in row.items():
                        # Try to convert to float
                        try:
                            metrics[key] = float(value)
                        except (ValueError, TypeError):
                            metrics[key] = value
    except Exception as e:
        logger.error(f"Error parsing CSV {csv_path}: {e}")
        metrics["parse_error"] = str(e)
    return metrics



def parse_metrics_file(file_path: str) -> dict:
    """Parse a VLMEvalKit metrics file (*_acc.csv or *_score.json) into metrics dict."""
    metrics = {}
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    # Skip nested dicts
                    if isinstance(value, dict):
                        continue
                    # Try to convert to float
                    if isinstance(value, (int, float)):
                        metrics[key] = float(value)
                    elif isinstance(value, str):
                        try:
                            metrics[key] = float(value)
                        except (ValueError, TypeError):
                            metrics[key] = value
                    else:
                        metrics[key] = value
        else:
            # CSV parsing (existing logic)
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    for row in rows:
                        for key, value in row.items():
                            try:
                                metrics[key] = float(value)
                            except (ValueError, TypeError):
                                metrics[key] = value
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        metrics["parse_error"] = str(e)
    
    return metrics



def parse_metrics(
    run_dir: str, 
    model_name: str, 
    task_name: str,
    metric_suffix: str = "_acc.csv"
) -> dict:
    """
    Parse metrics from VLMEvalKit output directory.
    
    Looks for files matching: {model}_{task}{suffix}
    e.g., InternVL2_5-2B_ChartQA_TEST_acc.csv
    """
    metrics = {
        "artifacts_dir": run_dir,
        "parsed_at": datetime.utcnow().isoformat()
    }
    
    # Primary metric file pattern
    primary_pattern = f"{model_name}_{task_name}{metric_suffix}"
    primary_path = os.path.join(run_dir, primary_pattern)
    
    if os.path.exists(primary_path):
        logger.info(f"Found primary metric file: {primary_path}")
        metrics.update(parse_metrics_file(primary_path))
    else:
        # Try to find any matching acc.csv file
        logger.warning(f"Primary metric file not found: {primary_path}")
        pattern = os.path.join(run_dir, f"*{task_name}*_acc.csv")
        matches = glob.glob(pattern)
        if matches:
            logger.info(f"Found alternative metric file: {matches[0]}")
            metrics.update(parse_metrics_file(matches[0]))
        else:
            # Last resort: find any acc.csv
            pattern = os.path.join(run_dir, "*_acc.csv")
            matches = glob.glob(pattern)
            if matches:
                logger.info(f"Found fallback metric file: {matches[0]}")
                metrics.update(parse_metrics_file(matches[0]))
            else:
                metrics["warning"] = "No metric CSV file found"
    
    # List all artifacts
    artifacts = []
    for f in os.listdir(run_dir):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath):
            artifacts.append({
                "name": f,
                "size": os.path.getsize(fpath),
                "type": Path(f).suffix
            })
    metrics["artifacts"] = artifacts
    
    return metrics


def run_vlmeval(
    task_vlmeval_data: str,
    model_vlmeval_model: str,
    extra_args: dict = None
) -> tuple[subprocess.CompletedProcess, str]:
    """
    Run VLMEvalKit evaluation.
    
    Returns (process_result, run_dir)
    """
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        VLMEVAL_RUN_SCRIPT,
        "--data", task_vlmeval_data,
        "--model", model_vlmeval_model
    ]
    
    # Add any extra arguments from config
    if extra_args:
        for key, value in extra_args.items():
            if key.startswith("--"):
                cmd.append(key)
            else:
                cmd.append(f"--{key}")
            if value is not None:
                cmd.append(str(value))
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Get output dir before running (to find the new run dir after)
    model_output_dir = os.path.join(VLMEVAL_OUTPUTS, model_vlmeval_model)
    existing_dirs = set(glob.glob(os.path.join(model_output_dir, "T*_G*")))
    
    # Run the evaluation
    result = subprocess.run(
        cmd,
        cwd=VLMEVAL_ROOT,
        capture_output=True,
        text=True,
        timeout=14400  # 2 hour timeout per eval
    )
    
    # Find the new run directory
    all_dirs = set(glob.glob(os.path.join(model_output_dir, "T*_G*")))
    new_dirs = all_dirs - existing_dirs
    
    if new_dirs:
        run_dir = max(new_dirs, key=os.path.getmtime)
    else:
        # Fallback to latest
        run_dir = find_latest_run_dir(model_output_dir)
    
    return result, run_dir


# =============================================================================
# Main Worker Loop
# =============================================================================

def process_one_run(db: EvalWorkerDB) -> bool:
    """
    Process a single queued eval run.
    Returns True if a run was processed, False if queue was empty.
    """
    run = db.get_next_queued_run()
    if not run:
        return False
    
    run_id = run["id"]
    task_vlmeval_data = run["vlmeval_data"]
    model_vlmeval_model = run["vlmeval_model"]
    model_name = run["model_name"]
    task_name = run["task_name"]
    metric_suffix = run.get("primary_metric_suffix") or "_acc.csv"
    #config = run.get("config_snapshot") or {}
    config = {}
    
    logger.info(f"Processing run {run_id}: {model_name} on {task_name}")
    
    # Get git commit for reproducibility
    git_commit = get_git_commit(VLMEVAL_ROOT)
    
    # Build command string for logging
    cmd_str = f"python run.py --data {task_vlmeval_data} --model {model_vlmeval_model}"
    
    try:
        # Mark as running
        db.mark_running(
            run_id,
            artifacts_dir="pending",  # Will update after run
            command=cmd_str,
            git_commit=git_commit
        )
        
        # Execute VLMEvalKit
        result, run_dir = run_vlmeval(
            task_vlmeval_data,
            model_vlmeval_model,
            extra_args=config.get("extra_args") if isinstance(config, dict) else None
        )
        
        # Save logs
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "worker_stdout.log"), "w") as f:
                f.write(result.stdout or "")
            with open(os.path.join(run_dir, "worker_stderr.log"), "w") as f:
                f.write(result.stderr or "")
        
        # Check for success
        if result.returncode != 0:
            error_msg = f"VLMEvalKit exited with code {result.returncode}\n"
            error_msg += f"stderr: {result.stderr[:2000] if result.stderr else 'N/A'}"
            logger.error(f"Run {run_id} failed: {error_msg}")
            db.mark_failed(run_id, error_msg)
            return True
        
        # Parse metrics
        if run_dir:
            metrics = parse_metrics(run_dir, model_vlmeval_model, task_vlmeval_data, metric_suffix)
            # Update artifacts_dir with actual path
            metrics["artifacts_dir"] = run_dir
        else:
            metrics = {"warning": "Could not determine run directory"}
        
        logger.info(f"Run {run_id} completed with metrics: {metrics}")
        db.mark_completed(run_id, metrics)
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Evaluation timed out after {e.timeout} seconds"
        logger.error(f"Run {run_id} timed out: {error_msg}")
        db.mark_failed(run_id, error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception(f"Run {run_id} failed with exception")
        db.mark_failed(run_id, error_msg)
    
    return True


def main():
    """Main worker loop."""
    logger.info("Starting eval worker...")
    logger.info(f"VLMEvalKit root: {VLMEVAL_ROOT}")
    logger.info(f"Database URL: {DATABASE_URL.split('@')[-1]}")  # Log without password
    
    # Verify VLMEvalKit exists
    if not os.path.exists(VLMEVAL_RUN_SCRIPT):
        logger.error(f"VLMEvalKit run.py not found at: {VLMEVAL_RUN_SCRIPT}")
        sys.exit(1)
    
    db = EvalWorkerDB(DATABASE_URL)
    
    consecutive_empty = 0
    
    while True:
        try:
            processed = process_one_run(db)
            
            if processed:
                consecutive_empty = 0
            elif consecutive_empty > 10:
                logger.info("Over 10 consectutive empty queues")
                break
            else:
                consecutive_empty += 1
                # Backoff when queue is empty
                sleep_time = min(POLL_INTERVAL_SECONDS * consecutive_empty, 60)
                time.sleep(sleep_time)
                 
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.exception("Unexpected error in worker loop")
            time.sleep(POLL_INTERVAL_SECONDS * 2)


if __name__ == "__main__":
    main()
