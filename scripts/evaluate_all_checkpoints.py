"""
Evaluate all model checkpoints on feedback prediction task.

Usage:
    python scripts/evaluate_all_checkpoints.py \
        --test-data feedback_data/feedback_v1/test.jsonl \
        --checkpoints-dir checkpoints/ \
        --output-dir eval_results/checkpoints/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "InternVL" / "internvl_chat"))

import argparse
import json
import subprocess
import os
from datetime import datetime

# Checkpoint configurations
# Maps checkpoint name patterns to their base model
CHECKPOINT_CONFIG = {
    # 8B models
    "8b": "OpenGVLab/InternVL2_5-8B",
    # Everything else defaults to 2B
    "default": "OpenGVLab/InternVL2_5-2B",
}

def get_base_model(checkpoint_name: str) -> str:
    """Determine base model from checkpoint name."""
    if "8b" in checkpoint_name.lower():
        return CHECKPOINT_CONFIG["8b"]
    return CHECKPOINT_CONFIG["default"]


def is_lora_checkpoint(checkpoint_path: Path) -> bool:
    """Check if checkpoint is LoRA (has adapter_config.json)."""
    return (checkpoint_path / "adapter_config.json").exists()


def is_vpt_checkpoint(checkpoint_name: str) -> bool:
    """Check if checkpoint is VPT."""
    return "vpt" in checkpoint_name.lower()


def find_best_subfolder(checkpoint_path: Path) -> Path:
    """Find the best checkpoint subfolder (best > final > epoch_X > root)."""
    if (checkpoint_path / "best").exists():
        return checkpoint_path / "best"
    if (checkpoint_path / "final").exists():
        return checkpoint_path / "final"
    
    # Look for epoch folders
    epoch_folders = sorted(
        [f for f in checkpoint_path.iterdir() if f.is_dir() and f.name.startswith("epoch_")],
        key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
        reverse=True
    )
    if epoch_folders:
        return epoch_folders[0]
    
    return checkpoint_path


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_data: str,
    output_dir: Path,
    base_model: str,
    eval_script: str,
    use_llm_judge: bool = True,
) -> dict:
    """Evaluate a single checkpoint."""
    
    checkpoint_name = checkpoint_path.name
    actual_path = find_best_subfolder(checkpoint_path)
    is_lora = is_lora_checkpoint(actual_path)
    is_vpt = is_vpt_checkpoint(checkpoint_name)
    
    output_file = output_dir / f"{checkpoint_name}.json"
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_name}")
    print(f"  Path: {actual_path}")
    print(f"  Base model: {base_model}")
    print(f"  Type: {'LoRA' if is_lora else 'VPT' if is_vpt else 'Full'}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", eval_script,
        "--test-data", test_data,
        "--base-model", base_model,
        "--output", str(output_file),
    ]
    
    if is_lora:
        cmd.extend(["--model", base_model, "--lora-path", str(actual_path)])
    elif is_vpt:
        cmd.extend(["--model", base_model, "--vpt-path", str(actual_path)])
    else:
        # Full model or base
        cmd.extend(["--model", str(actual_path)])
    
    if use_llm_judge:
        cmd.append("--use-llm-judge")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return {"checkpoint": checkpoint_name, "status": "error", "error": result.stderr}
        
        print(result.stdout)
        
        # Load results
        if output_file.exists():
            with open(output_file) as f:
                return {"checkpoint": checkpoint_name, "status": "success", "results": json.load(f)}
        else:
            return {"checkpoint": checkpoint_name, "status": "error", "error": "No output file created"}
            
    except subprocess.TimeoutExpired:
        return {"checkpoint": checkpoint_name, "status": "timeout"}
    except Exception as e:
        return {"checkpoint": checkpoint_name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints/")
    parser.add_argument("--output-dir", default="eval_results/checkpoints/")
    parser.add_argument("--eval-script", default="scripts/evaluate_model_feedback.py")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--filter", default=None, help="Only evaluate checkpoints matching this pattern")
    parser.add_argument("--exclude", default=None, help="Exclude checkpoints matching this pattern")
    parser.add_argument("--include-base", action="store_true", help="Also evaluate base models")
    
    args = parser.parse_args()
    
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all checkpoints
    checkpoints = sorted([
        d for d in checkpoints_dir.iterdir() 
        if d.is_dir() and not d.name.startswith(".")
    ])
    
    # Apply filters
    if args.filter:
        checkpoints = [c for c in checkpoints if args.filter in c.name]
    if args.exclude:
        checkpoints = [c for c in checkpoints if args.exclude not in c.name]
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate:")
    for c in checkpoints:
        print(f"  - {c.name}")
    
    # Track results
    all_results = []
    summary = []
    
    # Evaluate base models first if requested
    if args.include_base:
        base_models = ["OpenGVLab/InternVL2_5-2B"]
        for base in base_models:
            model_name = base.split("/")[-1]
            output_file = output_dir / f"base_{model_name}.json"
            
            cmd = [
                "python", args.eval_script,
                "--test-data", args.test_data,
                "--model", base,
                "--output", str(output_file),
            ]
            if args.use_llm_judge:
                cmd.append("--use-llm-judge")
            
            print(f"\nEvaluating base model: {base}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                    summary.append({
                        "checkpoint": f"base_{model_name}",
                        "type": "base",
                        **data.get("aggregate", {}),
                    })
    
    # Evaluate each checkpoint
    for checkpoint_path in checkpoints:
        base_model = get_base_model(checkpoint_path.name)
        
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            test_data=args.test_data,
            output_dir=output_dir,
            base_model=base_model,
            eval_script=args.eval_script,
            use_llm_judge=args.use_llm_judge,
        )
        
        all_results.append(result)
        
        # Extract summary
        if result["status"] == "success" and "results" in result:
            agg = result["results"].get("aggregate", {})
            checkpoint_type = "lora" if is_lora_checkpoint(find_best_subfolder(checkpoint_path)) else \
                             "vpt" if is_vpt_checkpoint(checkpoint_path.name) else "full"
            summary.append({
                "checkpoint": checkpoint_path.name,
                "type": checkpoint_type,
                "base_model": base_model,
                **agg,
            })
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_data": args.test_data,
            "num_checkpoints": len(checkpoints),
            "results": summary,
        }, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    
    # Sort by overall score if available
    summary_sorted = sorted(
        summary,
        key=lambda x: x.get("llm_judge", {}).get("overall", 0) if isinstance(x.get("llm_judge"), dict) 
                      else x.get("semantic_similarity", 0),
        reverse=True
    )
    
    print(f"\n{'Checkpoint':<45} {'Type':<8} {'Sem.Sim':<10} {'Judge':<10}")
    print("-" * 80)
    
    for s in summary_sorted:
        sem_sim = s.get("semantic_similarity", 0)
        judge = s.get("llm_judge", {})
        judge_overall = judge.get("llm_score", 0) if isinstance(judge, dict) else 0
        
        print(f"{s['checkpoint']:<45} {s.get('type', 'unk'):<8} {sem_sim:>8.2%}   {judge_overall:>8.2%}")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Individual results in: {output_dir}/")


if __name__ == "__main__":
    main()