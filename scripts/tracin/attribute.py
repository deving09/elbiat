"""
Compute TracIn influence scores.

influence(z_train, benchmark) = ∇L(z_train) · mean(∇L(z_test) for z_test in benchmark)
"""

import numpy as np
from pathlib import Path
from typing import Optional
import json


def load_gradients(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load gradients from npz file."""
    data = np.load(path)
    return data["gradients"], data["example_ids"]


def compute_influence_scores(
    train_gradients_path: str,
    test_gradients_path: str,
    output_path: str,
    normalize: bool = True,
    approach: str = 'every'
) -> dict:
    """
    Compute influence of each training example on test set performance.
    
    Args:
        train_gradients_path: Path to training gradients .npz
        test_gradients_path: Path to test gradients .npz
        output_path: Path to save influence scores
        normalize: Whether to normalize gradients before dot product
    
    Returns:
        Dict mapping train_example_id -> influence_score
    """
    print(f"Loading training gradients: {train_gradients_path}")
    train_grads, train_ids = load_gradients(train_gradients_path)
    
    print(f"Loading test gradients: {test_gradients_path}")
    test_grads, test_ids = load_gradients(test_gradients_path)
    
    print(f"Train: {train_grads.shape}, Test: {test_grads.shape}")
    
    # Normalize if requested
    if normalize:
        train_norms = np.linalg.norm(train_grads, axis=1, keepdims=True)
        train_norms = np.where(train_norms > 0, train_norms, 1)
        train_grads = train_grads / train_norms
        
        test_norms = np.linalg.norm(test_grads, axis=1, keepdims=True)
        test_norms = np.where(test_norms > 0, test_norms, 1)
        test_grads = test_grads / test_norms
    
    if approach == "mean":
        # Compute mean test gradient
        mean_test_grad = test_grads.mean(axis=0)
        
        # Compute influence scores (dot product with mean test gradient)
        influence_scores = train_grads @ mean_test_grad
    else:
        influence_scores = train_grads @ test_grads.T
        print(influence_scores.shape)
        influence_scores = influence_scores.mean(axis=1)
        print(influence_scores.shape)
    
    # Build result
    results = {
        "train_ids": [int(x) for x in train_ids.tolist()],
        "influence_scores": influence_scores.tolist(),
        "stats": {
            "mean": float(influence_scores.mean()),
            "std": float(influence_scores.std()),
            "min": float(influence_scores.min()),
            "max": float(influence_scores.max()),
            "n_positive": int((influence_scores > 0).sum()),
            "n_negative": int((influence_scores < 0).sum()),
        }
    }
    
    # Sort by influence
    sorted_indices = np.argsort(influence_scores)[::-1]  # Descending
    results["ranked"] = [
        {"id": int(train_ids[i]), "score": float(influence_scores[i])}
        for i in sorted_indices
    ]
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved influence scores to {output_path}")
    print(f"Stats: {results['stats']}")
    
    return results


def compute_multi_benchmark_attribution(
    train_gradients_path: str,
    benchmark_gradient_paths: dict[str, str],  # {benchmark_name: path}
    output_path: str,
) -> dict:
    """
    Compute influence scores for multiple benchmarks at once.
    
    Returns dict with per-benchmark scores for each training example.
    """
    print(f"Loading training gradients: {train_gradients_path}")
    train_grads, train_ids = load_gradients(train_gradients_path)
    
    # Normalize training gradients once
    train_norms = np.linalg.norm(train_grads, axis=1, keepdims=True)
    train_norms = np.where(train_norms > 0, train_norms, 1)
    train_grads_normalized = train_grads / train_norms
    
    results = {
        #"train_ids": train_ids.tolist(),
        "train_ids": [int(x) for x in train_ids.tolist()],

        "benchmarks": {},
    }
    
    for benchmark, test_path in benchmark_gradient_paths.items():
        print(f"\nProcessing {benchmark}...")
        test_grads, test_ids = load_gradients(test_path)
        
        # Normalize
        test_norms = np.linalg.norm(test_grads, axis=1, keepdims=True)
        test_norms = np.where(test_norms > 0, test_norms, 1)
        test_grads_normalized = test_grads / test_norms
        
        # Mean test gradient
        mean_test_grad = test_grads_normalized.mean(axis=0)
        
        # Influence scores
        scores = train_grads_normalized @ mean_test_grad
        
        results["benchmarks"][benchmark] = {
            "scores": scores.tolist(),
            "stats": {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "n_positive": int((scores > 0).sum()),
                "n_negative": int((scores < 0).sum()),
            }
        }
        
        print(f"  {benchmark}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    # Compute aggregate ranking
    all_scores = np.zeros(len(train_ids))
    for benchmark, data in results["benchmarks"].items():
        # Z-score normalize per benchmark before aggregating
        scores = np.array(data["scores"])
        z_scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        all_scores += z_scores
    
    all_scores /= len(benchmark_gradient_paths)
    results["aggregate_scores"] = all_scores.tolist()
    
    # Ranked by aggregate
    sorted_indices = np.argsort(all_scores)[::-1]
    results["aggregate_ranked"] = [
        {"id": int(train_ids[i]), "score": float(all_scores[i])}
        for i in sorted_indices
    ]

     
    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved multi-benchmark attribution to {output_path}")
    
    return results