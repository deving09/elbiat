#!/usr/bin/env python3
"""
Sync script to backfill eval_runs from existing VLMEvalKit outputs.

This scans your VLMEvalKit outputs directory and creates eval_run records
for any completed evaluations that aren't yet in the database.

Run with: python sync_vlmeval_outputs.py
"""

import os
import re
import csv
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/app"
)

VLMEVAL_OUTPUTS = os.getenv(
    "VLMEVAL_OUTPUTS",
    "/home/ubuntu/workspace/elbiat/external/VLMEvalKit/outputs"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_run_dir_name(dirname: str) -> Optional[dict]:
    """
    Parse a VLMEvalKit run directory name.
    Format: T<YYYYMMDD>_G<gitsha>
    Example: T20260204_G5f5146fe
    """
    match = re.match(r"T(\d{8})_G([a-f0-9]+)", dirname)
    if match:
        date_str, git_commit = match.groups()
        run_date = datetime.strptime(date_str, "%Y%m%d")
        return {
            "run_date": run_date,
            "git_commit": git_commit
        }
    return None


def parse_acc_csv(csv_path: str) -> dict:
    """Parse a VLMEvalKit *_acc.csv file."""
    metrics = {}
    try:
        with open(csv_path, 'r') as f:
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
                    print(f"key: {key}")
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


def find_metrics_for_task(run_dir: str, model_vlmeval: str, task_vlmeval: str, metric_suffix: str = "_acc.csv") -> Optional[dict]:
    """Find and parse metrics for a specific task in a run directory."""
    # Try exact match first
    pattern = f"{model_vlmeval}_{task_vlmeval}{metric_suffix}"
    exact_path = os.path.join(run_dir, pattern)
    
    if os.path.exists(exact_path):
        return parse_metrics_file(exact_path)
    
    # Try fuzzy match
    pattern = os.path.join(run_dir, f"*{task_vlmeval}*{metric_suffix}")
    matches = glob.glob(pattern)
    if matches:
        return parse_metrics_file(matches[0])
    
    return None


def get_artifacts_list(run_dir: str) -> list:
    """Get list of artifact files in a run directory."""
    artifacts = []
    if os.path.exists(run_dir):
        for f in os.listdir(run_dir):
            fpath = os.path.join(run_dir, f)
            if os.path.isfile(fpath):
                artifacts.append({
                    "name": f,
                    "size": os.path.getsize(fpath),
                    "type": Path(f).suffix
                })
    return artifacts


# =============================================================================
# Database Functions
# =============================================================================

def get_tasks(session) -> dict:
    """Get all tasks as a dict keyed by vlmeval_data."""
    result = session.execute(text("SELECT id, name, vlmeval_data, primary_metric_suffix FROM tasks"))
    tasks = {}
    for row in result:
        tasks[row.vlmeval_data] = {
            "id": row.id,
            "name": row.name,
            "vlmeval_data": row.vlmeval_data,
            "primary_metric_suffix": row.primary_metric_suffix
        }
    return tasks


def get_models(session) -> dict:
    """Get all models as a dict keyed by vlmeval_model."""
    result = session.execute(text("SELECT id, name, vlmeval_model FROM models"))
    models = {}
    for row in result:
        models[row.vlmeval_model] = {
            "id": row.id,
            "name": row.name,
            "vlmeval_model": row.vlmeval_model
        }
    return models


def run_exists(session, task_id: int, model_id: int, artifacts_dir: str) -> bool:
    """Check if a run already exists."""
    result = session.execute(
        text("""
            SELECT id FROM eval_runs 
            WHERE task_id = :task_id 
            AND model_id = :model_id 
            AND artifacts_dir = :artifacts_dir
        """),
        {"task_id": task_id, "model_id": model_id, "artifacts_dir": artifacts_dir}
    )
    return result.fetchone() is not None


def create_run(session, task_id: int, model_id: int, metrics: dict, artifacts_dir: str, run_info: dict):
    """Create an eval_run record."""
    session.execute(
        text("""
            INSERT INTO eval_runs 
            (task_id, model_id, status, metrics, artifacts_dir, git_commit, created_at, started_at, finished_at)
            VALUES 
            (:task_id, :model_id, 'COMPLETE', :metrics, :artifacts_dir, :git_commit, :created_at, :started_at, :finished_at)
        """),
        {
            "task_id": task_id,
            "model_id": model_id,
            "metrics": json.dumps(metrics),
            "artifacts_dir": artifacts_dir,
            "git_commit": run_info.get("git_commit"),
            "created_at": run_info.get("run_date"),
            "started_at": run_info.get("run_date"),
            "finished_at": run_info.get("run_date"),
        }
    )


# =============================================================================
# Main Sync Logic
# =============================================================================

def sync_outputs():
    """Main sync function."""
    print(f"Scanning VLMEvalKit outputs: {VLMEVAL_OUTPUTS}")
    
    if not os.path.exists(VLMEVAL_OUTPUTS):
        print(f"ERROR: Outputs directory not found: {VLMEVAL_OUTPUTS}")
        return
    
    session = SessionLocal()
    
    try:
        tasks = get_tasks(session)
        models = get_models(session)
        
        print(f"Found {len(tasks)} tasks in database")
        print(f"Found {len(models)} models in database")
        
        synced = 0
        skipped = 0
        errors = 0
        
        # Iterate over model directories
        for model_dir in os.listdir(VLMEVAL_OUTPUTS):
            model_path = os.path.join(VLMEVAL_OUTPUTS, model_dir)
            
            if not os.path.isdir(model_path):
                continue
            
            # Check if this model is registered
            model_vlmeval = model_dir
            if model_vlmeval not in models:
                print(f"  Skipping unregistered model: {model_vlmeval}")
                continue
            
            model_info = models[model_vlmeval]
            print(f"\nProcessing model: {model_vlmeval}")
            
            # Iterate over run directories (T*_G*)
            run_dirs = glob.glob(os.path.join(model_path, "T*_G*"))
            
            for run_dir in sorted(run_dirs):
                run_name = os.path.basename(run_dir)
                run_info = parse_run_dir_name(run_name)
                
                if not run_info:
                    print(f"  Could not parse run dir: {run_name}")
                    continue
                
                # Look for metric files for each known task
                for task_vlmeval, task_info in tasks.items():
                    metrics = find_metrics_for_task(
                        run_dir,
                        model_vlmeval,
                        task_vlmeval,
                        task_info["primary_metric_suffix"]
                    )
                    
                    if not metrics:
                        continue
                    
                    # Check if already synced
                    if run_exists(session, task_info["id"], model_info["id"], run_dir):
                        skipped += 1
                        continue
                    
                    # Add artifacts list to metrics
                    metrics["artifacts"] = get_artifacts_list(run_dir)
                    metrics["artifacts_dir"] = run_dir
                    
                    try:
                        create_run(
                            session,
                            task_info["id"],
                            model_info["id"],
                            metrics,
                            run_dir,
                            run_info
                        )
                        print(f"  ✓ Synced: {task_info['name']} - {run_name}")
                        synced += 1
                    except Exception as e:
                        print(f"  ✗ Error syncing {task_info['name']} - {run_name}: {e}")
                        errors += 1
        
        session.commit()
        
        print(f"\n{'='*50}")
        print(f"Sync complete!")
        print(f"  Synced: {synced}")
        print(f"  Skipped (already exists): {skipped}")
        print(f"  Errors: {errors}")
        
    finally:
        session.close()


def main():
    sync_outputs()


if __name__ == "__main__":
    main()
