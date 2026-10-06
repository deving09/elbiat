"""
Dump query logs for inspection.

Usage:
    python scripts/dump_query_logs.py
    python scripts/dump_query_logs.py --limit 10
    python scripts/dump_query_logs.py --user 1
    python scripts/dump_query_logs.py --export logs.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime

from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import Session

from app.models import QueryLog, User
from app.db import DATABASE_URL


def dump_logs(limit=20, user_id=None, export_path=None):
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
        query = select(QueryLog).order_by(desc(QueryLog.created_at))
        
        if user_id:
            query = query.where(QueryLog.user_id == user_id)
        
        query = query.limit(limit)
        
        logs = session.execute(query).scalars().all()
        
        print(f"\n{'='*60}")
        print(f"Query Logs (showing {len(logs)} of last {limit})")
        print(f"{'='*60}\n")
        
        results = []
        for log in logs:
            # Get username
            #user = session.get(User, log.user_id)
            username = log.user_id #user.email if user else f"User #{log.user_id}"
            
            result = {
                "id": log.id,
                "user": username,
                "image_id": log.image_id,
                "prompt": log.prompt,
                "response": log.response,
                "model": log.model_name,
                "latency_ms": log.latency_ms,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            results.append(result)
            
            # Print to console
            print(f"[{log.id}] {log.created_at}")
            print(f"    User: {username}")
            print(f"    Image: {log.image_id}")
            print(f"    Model: {log.model_name} ({log.latency_ms}ms)")
            print(f"    Prompt: {log.prompt[:200]}{'...' if len(log.prompt) > 200 else ''}")
            print(f"    Response: {log.response[:200]}{'...' if len(log.response) > 200 else ''}")
            print()
        
        # Export if requested
        if export_path:
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Exported to {export_path}")
        
        # Summary stats
        total = session.query(QueryLog).count()
        print(f"{'='*60}")
        print(f"Total logs in DB: {total}")
        
        if logs:
            avg_latency = sum(l.latency_ms or 0 for l in logs) / len(logs)
            print(f"Avg latency (shown): {avg_latency:.0f}ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', '-n', type=int, default=20)
    parser.add_argument('--user', '-u', type=int, default=None)
    parser.add_argument('--export', '-e', type=str, default=None)
    
    args = parser.parse_args()
    dump_logs(args.limit, args.user, args.export)


if __name__ == '__main__':
    main()