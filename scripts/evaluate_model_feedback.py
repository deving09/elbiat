"""
Evaluate a VLM model on test data with multiple metrics.

Usage:
    python scripts/evaluate_model.py \
        --test-data feedback_data/dpo_v1/test.jsonl \
        --model InternVL2_5-2B-DPO-v1 \
        --output eval_results/dpo_v1.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "InternVL" / "internvl_chat"))

import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess


@dataclass
class MetricResult:
    exact_match: float
    contains_match: float
    numeric_precision: float
    numeric_recall: float
    numeric_f1: float
    term_precision: float
    term_recall: float
    term_f1: float
    semantic_similarity: float
    
    def overall(self) -> float:
        """Weighted average of metrics."""
        return (
            0.1 * self.exact_match +
            0.1 * self.contains_match +
            0.3 * self.numeric_f1 +
            0.2 * self.term_f1 +
            0.3 * self.semantic_similarity
        )


def extract_numbers(text: str) -> list[float]:
    pattern = r'[-+]?\d*\.?\d+%?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            if m.endswith('%'):
                numbers.append(float(m[:-1]))
            else:
                numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def numeric_f1(prediction: str, reference: str, tolerance: float = 0.5) -> tuple[float, float, float]:
    pred_nums = extract_numbers(prediction)
    ref_nums = extract_numbers(reference)
    
    if not ref_nums:
        return 1.0, 1.0, 1.0
    if not pred_nums:
        return 0.0, 0.0, 0.0
    
    matched_ref = set()
    matched_pred = set()
    
    for i, p in enumerate(pred_nums):
        for j, r in enumerate(ref_nums):
            if j not in matched_ref and abs(p - r) <= tolerance:
                matched_pred.add(i)
                matched_ref.add(j)
                break
    
    precision = len(matched_pred) / len(pred_nums) if pred_nums else 0
    recall = len(matched_ref) / len(ref_nums) if ref_nums else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def extract_key_terms(text: str) -> set[str]:
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall',
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
        'and', 'or', 'but', 'not', 'no', 'yes', 'that', 'this',
        'it', 'its', 'as', 'if', 'than', 'so', 'what', 'which',
        'who', 'how', 'when', 'where', 'why', 'all', 'each',
        'only', 'also', 'just', 'more', 'most', 'other', 'some',
        'such', 'too', 'very', 'can', 'percent', 'percentage',
    }
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {w for w in words if w not in stop_words and len(w) > 2}


def term_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_terms = extract_key_terms(prediction)
    ref_terms = extract_key_terms(reference)
    
    if not ref_terms:
        return 1.0, 1.0, 1.0
    
    overlap = pred_terms & ref_terms
    
    precision = len(overlap) / len(pred_terms) if pred_terms else 0
    recall = len(overlap) / len(ref_terms) if ref_terms else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


class Evaluator:
    def __init__(
        self,
        model_path: str,
        root_dir: str = "/home/ubuntu/workspace/elbiat",
        use_semantic: bool = True,
    ):
        self.root_dir = Path(root_dir)
        
        # Load VLM
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        
        self.transform = build_transform(is_train=False, input_size=448)
        
        # Semantic similarity model
        if use_semantic:
            print("Loading semantic similarity model...")
            self.sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.sem_model = None
    
    def load_image(self, image_path: str) -> torch.Tensor:
        full_path = self.root_dir / image_path
        image = Image.open(full_path).convert("RGB")
        images = dynamic_preprocess(image, image_size=448, max_num=6)
        pixel_values = torch.stack([self.transform(img) for img in images])
        return pixel_values.to(self.model.device, dtype=torch.bfloat16)
    
    def generate(self, image_path: str, question: str) -> str:
        pixel_values = self.load_image(image_path)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            dict(max_new_tokens=256, do_sample=False),
        )
        return response
    
    def compute_metrics(self, prediction: str, reference: str) -> MetricResult:
        # Basic matches
        exact = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
        contains = 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0
        
        # Numeric F1
        num_p, num_r, num_f1 = numeric_f1(prediction, reference)
        
        # Term F1
        term_p, term_r, t_f1 = term_f1(prediction, reference)
        
        # Semantic similarity
        if self.sem_model:
            embeddings = self.sem_model.encode([prediction, reference])
            sem_sim = float(np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            ))
        else:
            sem_sim = 0.0
        
        return MetricResult(
            exact_match=exact,
            contains_match=contains,
            numeric_precision=num_p,
            numeric_recall=num_r,
            numeric_f1=num_f1,
            term_precision=term_p,
            term_recall=term_r,
            term_f1=t_f1,
            semantic_similarity=sem_sim,
        )
    
    def evaluate_dataset(self, test_data: list[dict]) -> dict:
        results = []
        
        for sample in tqdm(test_data, desc="Evaluating"):
            # Parse sample
            question = ""
            original_answer = ""
            reference = ""
            
            for turn in sample.get("conversations", []):
                if turn["from"] == "human":
                    # Extract just the question
                    text = turn["value"].replace("<image>\n", "")
                    if "Question:" in text:
                        question = text #text.split("Question:")[-1].split("\n")[0].strip()
                elif turn["from"] == "gpt":
                    reference = turn["value"].strip()
            
            if not question or not reference:
                continue
            
            # Generate prediction
            prediction = self.generate(sample["image"], question)
            
            # Compute metrics
            metrics = self.compute_metrics(prediction, reference)

            #print(text)
            print(question)
            print(prediction)
            print(reference)

            print(asdict(metrics))
            #1/0

            
            results.append({
                "id": sample.get("id"),
                "image": sample["image"],
                "question": question,
                "prediction": prediction,
                "reference": reference,
                "metrics": asdict(metrics),
                "overall": metrics.overall(),
            })
        
        # Aggregate
        if not results:
            return {"error": "No valid samples"}
        
        aggregate = {
            "num_samples": len(results),
            "exact_match": np.mean([r["metrics"]["exact_match"] for r in results]),
            "contains_match": np.mean([r["metrics"]["contains_match"] for r in results]),
            "numeric_f1": np.mean([r["metrics"]["numeric_f1"] for r in results]),
            "term_f1": np.mean([r["metrics"]["term_f1"] for r in results]),
            "semantic_similarity": np.mean([r["metrics"]["semantic_similarity"] for r in results]),
            "overall": np.mean([r["overall"] for r in results]),
        }
        
        return {
            "aggregate": aggregate,
            "per_sample": results,
        }


def load_test_data(jsonl_path: str) -> list[dict]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--model", default="OpenGVLab/InternVL2_5-2B")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--root-dir", default="/home/ubuntu/workspace/elbiat")
    
    args = parser.parse_args()
    
    # Load data
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    evaluator = Evaluator(args.model, args.root_dir)
    results = evaluator.evaluate_dataset(test_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    agg = results["aggregate"]
    print(f"Samples: {agg['num_samples']}")
    print(f"Exact Match: {agg['exact_match']:.2%}")
    print(f"Contains Match: {agg['contains_match']:.2%}")
    print(f"Numeric F1: {agg['numeric_f1']:.2%}")
    print(f"Term F1: {agg['term_f1']:.2%}")
    print(f"Semantic Similarity: {agg['semantic_similarity']:.2%}")
    print(f"Overall: {agg['overall']:.2%}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
