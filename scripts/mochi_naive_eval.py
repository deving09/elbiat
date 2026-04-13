"""
MOCHI Naive Evaluator for VLMs.

This evaluator uses the "naive" approach where multiple images are passed
separately to the VLM model. This tests the model's ability to handle
multiple images in a single context.

Usage:
    python mochi_naive_eval.py \
        --model OpenGVLab/InternVL2_5-2B \
        --output eval_results/mochi_naive.json \
        --max-samples 100
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

# Add paths for InternVL
sys.path.insert(0, str(Path(__file__).parent))

from mochi_dataset import MOCHIDataset, parse_answer, INDEX_TO_LETTER


@dataclass
class EvalResult:
    """Result for a single evaluation trial."""
    trial_id: str
    n_objects: int
    correct_answer: str
    predicted_answer: Optional[str]
    raw_response: str
    correct: bool
    human_accuracy: float
    condition: str
    dataset: str


@dataclass
class EvalSummary:
    """Summary statistics for evaluation."""
    total: int
    correct: int
    accuracy: float
    accuracy_3obj: float
    accuracy_4obj: float
    by_condition: dict
    by_dataset: dict
    human_alignment: float  # Correlation with human accuracy


class MOCHINaiveEvaluator:
    """
    Evaluator for MOCHI benchmark using naive multi-image approach.
    
    This evaluator loads multiple images and passes them to the VLM
    in a single query, testing multi-image understanding.
    """
    
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ):
        """
        Initialize evaluator with a VLM model.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            lora_path: Optional path to LoRA weights
            device: Device to run on
            max_new_tokens: Max tokens to generate
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading model: {model_path}")
        
        # Import InternVL components
        try:
            from internvl.model.internvl_chat import InternVLChatModel
            from internvl.train.dataset import build_transform, dynamic_preprocess
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
            
            self.model = InternVLChatModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            
            # Load LoRA if specified
            if lora_path:
                print(f"Loading LoRA weights: {lora_path}")
                from peft import PeftModel
                self.model.language_model = PeftModel.from_pretrained(
                    self.model.language_model, lora_path
                )
                self.model.language_model = self.model.language_model.merge_and_unload()
            
            self.model = self.model.eval()
            self.transform = build_transform(is_train=False, input_size=448)
            self.dynamic_preprocess = dynamic_preprocess
            self.model_type = "internvl"
            
        except ImportError:
            print("InternVL not available, using HuggingFace transformers...")
            # Fallback to generic multi-modal model loading
            from transformers import AutoModel, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            self.model_type = "hf"
        
        print(f"Model loaded: {self.model_type}")
    
    def _preprocess_images_internvl(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess multiple images for InternVL."""
        all_pixel_values = []
        
        for img in images:
            # Apply dynamic preprocessing (splits into multiple patches)
            processed = self.dynamic_preprocess(img, image_size=448, max_num=6)
            pixel_values = torch.stack([self.transform(p) for p in processed])
            all_pixel_values.append(pixel_values)
        
        # Concatenate all image patches
        combined = torch.cat(all_pixel_values, dim=0)
        return combined.to(self.model.device, dtype=torch.bfloat16)
    
    def _build_multi_image_prompt(self, prompt: str, n_images: int) -> str:
        """Build prompt with image placeholders for multi-image input."""
        # InternVL uses <image> token for each image
        image_tokens = " ".join([f"<image>" for _ in range(n_images)])
        return f"{image_tokens}\n{prompt}"
    
    def generate(self, images: list[Image.Image], prompt: str) -> str:
        """
        Generate response for multiple images.
        
        Args:
            images: List of PIL images
            prompt: Text prompt
        
        Returns:
            Model's response text
        """
        if self.model_type == "internvl":
            pixel_values = self._preprocess_images_internvl(images)
            full_prompt = self._build_multi_image_prompt(prompt, len(images))
            
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_prompt,
                    dict(max_new_tokens=self.max_new_tokens, do_sample=False),
                )
            return response
        
        else:
            # Generic HF model
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            return response
    
    def evaluate_trial(self, trial: dict) -> EvalResult:
        """Evaluate a single MOCHI trial."""
        images = trial["images"]
        prompt = trial["prompt"]
        
        # Generate response
        response = self.generate(images, prompt)
        
        # Parse answer
        predicted = parse_answer(response, trial["n_objects"])
        correct = predicted == trial["answer"]
        
        return EvalResult(
            trial_id=trial["id"],
            n_objects=trial["n_objects"],
            correct_answer=trial["answer"],
            predicted_answer=predicted,
            raw_response=response,
            correct=correct,
            human_accuracy=trial["metadata"]["human_avg"],
            condition=trial["metadata"]["condition"],
            dataset=trial["metadata"]["dataset"],
        )
    
    def evaluate(
        self,
        dataset: MOCHIDataset,
        output_path: Optional[str] = None,
    ) -> tuple[list[EvalResult], EvalSummary]:
        """
        Evaluate on entire MOCHI dataset.
        
        Args:
            dataset: MOCHI dataset instance
            output_path: Optional path to save results
        
        Returns:
            Tuple of (list of results, summary statistics)
        """
        results = []
        
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            trial = dataset[idx]
            result = self.evaluate_trial(trial)
            results.append(result)
        
        # Compute summary
        summary = self._compute_summary(results)
        
        # Save results
        if output_path:
            self._save_results(results, summary, output_path)
        
        return results, summary
    
    def _compute_summary(self, results: list[EvalResult]) -> EvalSummary:
        """Compute summary statistics."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        
        # By number of objects
        results_3obj = [r for r in results if r.n_objects == 3]
        results_4obj = [r for r in results if r.n_objects == 4]
        
        acc_3obj = sum(1 for r in results_3obj if r.correct) / len(results_3obj) if results_3obj else 0
        acc_4obj = sum(1 for r in results_4obj if r.correct) / len(results_4obj) if results_4obj else 0
        
        # By condition
        by_condition = {}
        conditions = set(r.condition for r in results)
        for cond in conditions:
            cond_results = [r for r in results if r.condition == cond]
            by_condition[cond] = {
                "total": len(cond_results),
                "correct": sum(1 for r in cond_results if r.correct),
                "accuracy": sum(1 for r in cond_results if r.correct) / len(cond_results),
            }
        
        # By dataset
        by_dataset = {}
        datasets = set(r.dataset for r in results)
        for ds in datasets:
            ds_results = [r for r in results if r.dataset == ds]
            by_dataset[ds] = {
                "total": len(ds_results),
                "correct": sum(1 for r in ds_results if r.correct),
                "accuracy": sum(1 for r in ds_results if r.correct) / len(ds_results),
            }
        
        # Human alignment (correlation with human accuracy)
        # Simple: accuracy on easy (human_avg > 0.8) vs hard (human_avg < 0.5) trials
        easy = [r for r in results if r.human_accuracy > 0.8]
        hard = [r for r in results if r.human_accuracy < 0.5]
        
        easy_acc = sum(1 for r in easy if r.correct) / len(easy) if easy else 0
        hard_acc = sum(1 for r in hard if r.correct) / len(hard) if hard else 0
        human_alignment = easy_acc - hard_acc  # Should be positive if aligned
        
        return EvalSummary(
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0,
            accuracy_3obj=acc_3obj,
            accuracy_4obj=acc_4obj,
            by_condition=by_condition,
            by_dataset=by_dataset,
            human_alignment=human_alignment,
        )
    
    def _save_results(
        self,
        results: list[EvalResult],
        summary: EvalSummary,
        output_path: str,
    ):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "model": self.model_path,
                "mode": "naive",
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(results),
            },
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        print(f"\n=== Summary ===")
        print(f"Total: {summary.total}")
        print(f"Correct: {summary.correct}")
        print(f"Accuracy: {summary.accuracy:.2%}")
        print(f"Accuracy (3 objects): {summary.accuracy_3obj:.2%}")
        print(f"Accuracy (4 objects): {summary.accuracy_4obj:.2%}")
        print(f"Human alignment: {summary.human_alignment:+.2f}")
        
        print(f"\n=== By Condition ===")
        for cond, stats in sorted(summary.by_condition.items()):
            print(f"  {cond}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(description="MOCHI Naive Evaluator")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA weights")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--subset", default=None, help="Filter by dataset (barense, shapegen, etc)")
    parser.add_argument("--condition", default=None, help="Filter by condition")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = MOCHIDataset(
        mode="naive",
        max_samples=args.max_samples,
        subset=args.subset,
        condition=args.condition,
    )
    
    # Create evaluator
    evaluator = MOCHINaiveEvaluator(
        model_path=args.model,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Run evaluation
    results, summary = evaluator.evaluate(dataset, args.output)
    
    print(f"\nEvaluation complete!")
    print(f"Overall accuracy: {summary.accuracy:.2%}")


if __name__ == "__main__":
    main()
