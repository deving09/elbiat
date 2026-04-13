"""
Unified MOCHI Evaluation Script.

Supports two evaluation approaches:
1. naive: Pass multiple images separately to the VLM
2. grid: Merge images into a labeled 2x2 grid

Can be used standalone or integrated with VLMEvalKit.

Usage:
    # Standalone evaluation
    python run_mochi_eval.py \
        --model OpenGVLab/InternVL2_5-2B \
        --mode naive \
        --output eval_results/mochi_naive.json

    # Compare both approaches
    python run_mochi_eval.py \
        --model OpenGVLab/InternVL2_5-2B \
        --mode both \
        --output eval_results/mochi_comparison.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from mochi_dataset import MOCHIDataset
from mochi_naive_eval import MOCHINaiveEvaluator, EvalSummary
from mochi_grid_eval import MOCHIGridEvaluator


def compare_results(naive_summary: EvalSummary, grid_summary: EvalSummary) -> dict:
    """Compare results from both approaches."""
    comparison = {
        "overall": {
            "naive_accuracy": naive_summary.accuracy,
            "grid_accuracy": grid_summary.accuracy,
            "difference": grid_summary.accuracy - naive_summary.accuracy,
            "better": "grid" if grid_summary.accuracy > naive_summary.accuracy else "naive",
        },
        "by_n_objects": {
            "3_objects": {
                "naive": naive_summary.accuracy_3obj,
                "grid": grid_summary.accuracy_3obj,
                "diff": grid_summary.accuracy_3obj - naive_summary.accuracy_3obj,
            },
            "4_objects": {
                "naive": naive_summary.accuracy_4obj,
                "grid": grid_summary.accuracy_4obj,
                "diff": grid_summary.accuracy_4obj - naive_summary.accuracy_4obj,
            },
        },
        "human_alignment": {
            "naive": naive_summary.human_alignment,
            "grid": grid_summary.human_alignment,
        },
        "by_condition": {},
    }
    
    # Compare by condition
    all_conditions = set(naive_summary.by_condition.keys()) | set(grid_summary.by_condition.keys())
    for cond in all_conditions:
        naive_acc = naive_summary.by_condition.get(cond, {}).get("accuracy", 0)
        grid_acc = grid_summary.by_condition.get(cond, {}).get("accuracy", 0)
        comparison["by_condition"][cond] = {
            "naive": naive_acc,
            "grid": grid_acc,
            "diff": grid_acc - naive_acc,
        }
    
    return comparison


def print_comparison(comparison: dict):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("MOCHI EVALUATION COMPARISON: Naive vs Grid")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Naive':>12} {'Grid':>12} {'Diff':>12}")
    print("-" * 70)
    
    print(f"{'Overall Accuracy':<30} {comparison['overall']['naive_accuracy']:>11.2%} {comparison['overall']['grid_accuracy']:>11.2%} {comparison['overall']['difference']:>+11.2%}")
    print(f"{'3-Object Trials':<30} {comparison['by_n_objects']['3_objects']['naive']:>11.2%} {comparison['by_n_objects']['3_objects']['grid']:>11.2%} {comparison['by_n_objects']['3_objects']['diff']:>+11.2%}")
    print(f"{'4-Object Trials':<30} {comparison['by_n_objects']['4_objects']['naive']:>11.2%} {comparison['by_n_objects']['4_objects']['grid']:>11.2%} {comparison['by_n_objects']['4_objects']['diff']:>+11.2%}")
    print(f"{'Human Alignment':<30} {comparison['human_alignment']['naive']:>+11.2f} {comparison['human_alignment']['grid']:>+11.2f}")
    
    print(f"\n{'By Condition':<30}")
    print("-" * 70)
    for cond, stats in sorted(comparison["by_condition"].items()):
        print(f"  {cond:<28} {stats['naive']:>11.2%} {stats['grid']:>11.2%} {stats['diff']:>+11.2%}")
    
    print("\n" + "=" * 70)
    better = comparison["overall"]["better"]
    diff = abs(comparison["overall"]["difference"])
    print(f"Result: {better.upper()} approach is better by {diff:.2%}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MOCHI Unified Evaluator")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA weights")
    parser.add_argument("--mode", choices=["naive", "grid", "both"], default="both",
                        help="Evaluation mode")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--subset", default=None, help="Filter by dataset")
    parser.add_argument("--condition", default=None, help="Filter by condition")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=512)
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "metadata": {
            "model": args.model,
            "lora_path": args.lora_path,
            "mode": args.mode,
            "max_samples": args.max_samples,
            "subset": args.subset,
            "condition": args.condition,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Run naive evaluation
    if args.mode in ["naive", "both"]:
        print("\n" + "=" * 50)
        print("Running NAIVE evaluation (multi-image)")
        print("=" * 50)
        
        dataset_naive = MOCHIDataset(
            mode="naive",
            max_samples=args.max_samples,
            subset=args.subset,
            condition=args.condition,
        )
        
        evaluator_naive = MOCHINaiveEvaluator(
            model_path=args.model,
            lora_path=args.lora_path,
            max_new_tokens=args.max_new_tokens,
        )
        
        naive_results, naive_summary = evaluator_naive.evaluate(dataset_naive)
        
        results["naive"] = {
            "summary": {
                "total": naive_summary.total,
                "correct": naive_summary.correct,
                "accuracy": naive_summary.accuracy,
                "accuracy_3obj": naive_summary.accuracy_3obj,
                "accuracy_4obj": naive_summary.accuracy_4obj,
                "human_alignment": naive_summary.human_alignment,
                "by_condition": naive_summary.by_condition,
                "by_dataset": naive_summary.by_dataset,
            },
            "results": [
                {
                    "trial_id": r.trial_id,
                    "correct_answer": r.correct_answer,
                    "predicted_answer": r.predicted_answer,
                    "correct": r.correct,
                    "raw_response": r.raw_response[:200],  # Truncate for space
                }
                for r in naive_results
            ],
        }
        
        print(f"\nNaive accuracy: {naive_summary.accuracy:.2%}")
    
    # Run grid evaluation
    if args.mode in ["grid", "both"]:
        print("\n" + "=" * 50)
        print("Running GRID evaluation (single image)")
        print("=" * 50)
        
        dataset_grid = MOCHIDataset(
            mode="single_image",
            max_samples=args.max_samples,
            subset=args.subset,
            condition=args.condition,
        )
        
        evaluator_grid = MOCHIGridEvaluator(
            model_path=args.model,
            lora_path=args.lora_path,
            max_new_tokens=args.max_new_tokens,
            grid_size=args.grid_size,
        )
        
        grid_results, grid_summary = evaluator_grid.evaluate(dataset_grid)
        
        results["grid"] = {
            "summary": {
                "total": grid_summary.total,
                "correct": grid_summary.correct,
                "accuracy": grid_summary.accuracy,
                "accuracy_3obj": grid_summary.accuracy_3obj,
                "accuracy_4obj": grid_summary.accuracy_4obj,
                "human_alignment": grid_summary.human_alignment,
                "by_condition": grid_summary.by_condition,
                "by_dataset": grid_summary.by_dataset,
            },
            "results": [
                {
                    "trial_id": r.trial_id,
                    "correct_answer": r.correct_answer,
                    "predicted_answer": r.predicted_answer,
                    "correct": r.correct,
                    "raw_response": r.raw_response[:200],
                }
                for r in grid_results
            ],
        }
        
        print(f"\nGrid accuracy: {grid_summary.accuracy:.2%}")
    
    # Compare if both
    if args.mode == "both":
        comparison = compare_results(naive_summary, grid_summary)
        results["comparison"] = comparison
        print_comparison(comparison)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
