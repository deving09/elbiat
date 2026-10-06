"""
Import TracIn attribution results into the database.

Usage:
    python scripts/tracin/import_to_db.py \
        --checkpoint feedback_v1 \
        --results tracin_results/chartqa_attribution.json \
        --benchmark chartqa

    # Or import multiple at once:
    python scripts/tracin/import_to_db.py \
        --checkpoint feedback_v1 \
        --results tracin_results/chartqa_attribution.json tracin_results/blink_attribution.json \
        --benchmark chartqa blink
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from app.db import get_db, engine
from app import models


def load_attribution_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def import_attribution(
    db: Session,
    checkpoint_id: str,
    benchmark: str,
    results: dict,
    method: str = "tracin",
):
    """Import attribution results for a single benchmark."""
    
    train_ids = results["train_ids"]
    scores = results["influence_scores"]
    
    # Compute z-scores
    scores_arr = np.array(scores)
    z_scores = stats.zscore(scores_arr)
    
    # Compute ranks (1 = highest positive influence)
    ranks = stats.rankdata(-scores_arr, method='ordinal')  # negative for descending
    
    print(f"Importing {len(train_ids)} attributions for {benchmark}...")
    
    # Build rows
    rows = []
    for i, (convo_id, score, z, rank) in enumerate(zip(train_ids, scores, z_scores, ranks)):
        rows.append({
            "convo_id": int(convo_id),
            "checkpoint_id": checkpoint_id,
            "benchmark": benchmark,
            "method": method,
            "influence_score": float(score),
            "z_score": float(z),
            "rank": int(rank),
            "computed_at": datetime.utcnow(),
        })
    
    # Upsert (insert or update on conflict)
    stmt = insert(models.DataAttribution).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_attribution_convo_checkpoint_benchmark_method",
        set_={
            "influence_score": stmt.excluded.influence_score,
            "z_score": stmt.excluded.z_score,
            "rank": stmt.excluded.rank,
            "computed_at": stmt.excluded.computed_at,
        }
    )
    
    db.execute(stmt)
    db.commit()
    
    print(f"  Imported {len(rows)} rows")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Top 3: {[r['convo_id'] for r in sorted(rows, key=lambda x: -x['influence_score'])[:3]]}")


def main():
    parser = argparse.ArgumentParser(description="Import TracIn results to database")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint ID (e.g., feedback_v1)")
    parser.add_argument("--results", nargs="+", required=True, help="Path(s) to attribution JSON files")
    parser.add_argument("--benchmark", nargs="+", required=True, help="Benchmark name(s) matching results files")
    parser.add_argument("--method", default="tracin", help="Attribution method name")
    
    args = parser.parse_args()
    
    if len(args.results) != len(args.benchmark):
        print("Error: Must provide same number of --results and --benchmark arguments")
        sys.exit(1)
    
    # Get database session
    with Session(engine) as db:
        for results_path, benchmark in zip(args.results, args.benchmark):
            print(f"\nLoading {results_path}...")
            results = load_attribution_json(results_path)
            
            import_attribution(
                db=db,
                checkpoint_id=args.checkpoint,
                benchmark=benchmark,
                results=results,
                method=args.method,
            )
    
    print("\nDone!")


if __name__ == "__main__":
    main()