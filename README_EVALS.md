# VLMEvalKit Evaluation System

This system provides a Task page for running and displaying VLMEvalKit benchmark results.

## Files Overview

| File | Purpose |
|------|---------|
| `models_eval.py` | SQLAlchemy models (Task, Model, EvalRun) |
| `migrations/001_create_eval_tables.py` | Alembic migration |
| `routes_evals.py` | FastAPI API endpoints |
| `worker_evals.py` | Background worker that runs VLMEvalKit |
| `seed_evals.py` | Populate initial tasks and models |
| `sync_vlmeval_outputs.py` | Backfill runs from existing outputs |
| `worker_evals.service` | Systemd service file |

## Setup Steps

### 1. Update Database Models

Copy the models from `models_eval.py` into your `app/models.py`:

```python
# In app/models.py, add:
from enum import Enum as PyEnum

class EvalStatus(str, PyEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ModelType(str, PyEnum):
    VLM = "vlm"
    LLM = "llm"

# Then add the Task, Model, and EvalRun classes
```

### 2. Run the Migration

Update the migration file's `down_revision` to point to your latest migration, then:

```bash
cd /home/ubuntu/workspace/elbiat
cp /path/to/migrations/001_create_eval_tables.py alembic/versions/

# Edit the file to set correct down_revision
alembic upgrade head
```

### 3. Seed Initial Data

Update `seed_evals.py` with your database URL, then:

```bash
# Set your database URL
export DATABASE_URL="postgresql://user:password@localhost:5432/elbiat"

python seed_evals.py
```

### 4. Add API Routes

Add the routes to your FastAPI app:

```python
# In app/main.py or wherever you configure routes:
from app.routes.evals import router as evals_router

app.include_router(evals_router)
```

### 5. Sync Existing Outputs (Optional)

If you already have VLMEvalKit outputs, sync them to the database:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/elbiat"
export VLMEVAL_OUTPUTS="/home/ubuntu/workspace/elbiat/external/VLMEvalKit/outputs"

python sync_vlmeval_outputs.py
```

### 6. Start the Worker

For development:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/elbiat"
export VLMEVAL_ROOT="/home/ubuntu/workspace/elbiat/external/VLMEvalKit"

python worker_evals.py
```

For production (systemd):
```bash
# Update the service file with your paths
sudo cp worker_evals.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable worker_evals
sudo systemctl start worker_evals
```

## API Endpoints

### Tasks

- `GET /api/evals/tasks` - List all tasks
- `GET /api/evals/tasks/{name}` - Get task details
- `POST /api/evals/tasks` - Create new task

### Models

- `GET /api/evals/models` - List all models
- `GET /api/evals/models/{name}` - Get model details
- `POST /api/evals/models` - Create new model

### Eval Runs

- `GET /api/evals/tasks/{task_name}/runs` - List runs for a task
- `GET /api/evals/tasks/{task_name}/leaderboard` - Get leaderboard
- `POST /api/evals/tasks/{task_name}/runs` - Trigger new evaluation
- `GET /api/evals/runs/{id}` - Get run details
- `GET /api/evals/runs/{id}/artifacts` - List run artifacts

## Triggering an Evaluation

```bash
curl -X POST http://localhost:8000/api/evals/tasks/charxiv/runs \
  -H "Content-Type: application/json" \
  -d '{"model_name": "internvl2_5_2b"}'
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgresql://... | PostgreSQL connection string |
| `VLMEVAL_ROOT` | /home/ubuntu/.../VLMEvalKit | Path to VLMEvalKit |
| `VLMEVAL_OUTPUTS` | {VLMEVAL_ROOT}/outputs | Where outputs are stored |

## Adding New Tasks

1. Find the VLMEvalKit `--data` name for the task
2. Run a test to find the metric file suffix (usually `_acc.csv`)
3. Add to database:

```python
# Via API
curl -X POST http://localhost:8000/api/evals/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "infographicvqa",
    "display_name": "InfographicVQA",
    "vlmeval_data": "InfoVQA",
    "primary_metric_suffix": "_acc.csv",
    "primary_metric_key": "acc"
  }'
```

## Adding New Models

1. Find the VLMEvalKit `--model` name
2. Add to database:

```python
curl -X POST http://localhost:8000/api/evals/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen2_vl_72b",
    "display_name": "Qwen2-VL 72B",
    "vlmeval_model": "Qwen2-VL-72B-Instruct",
    "model_type": "vlm",
    "params_b": 72.0
  }'
```

## Troubleshooting

### Worker not picking up jobs

1. Check worker logs: `tail -f worker_evals.log`
2. Verify database connection
3. Check for queued runs: `SELECT * FROM eval_runs WHERE status = 'queued'`

### Metrics not parsing

1. Check the artifacts directory for the expected file
2. Verify the `primary_metric_suffix` matches
3. Check worker logs for parse errors

### VLMEvalKit errors

1. Check `worker_stdout.log` and `worker_stderr.log` in artifacts dir
2. Verify CUDA/GPU availability
3. Check VLMEvalKit dependencies
