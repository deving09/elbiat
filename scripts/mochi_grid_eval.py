"""
MOCHI Single Image Grid Evaluator for VLMs.

This evaluator uses the "single image" approach where multiple images are 
merged into a 2x2 grid with labels A, B, C, D. This tests the model's 
ability to understand spatial layouts and labeled regions.

Usage:
    python mochi_grid_eval.py \
        --model OpenGVLab/InternVL2_5-2B \
        --output eval_results/mochi_grid.json \
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

sys.path.insert(0, str(Path(__file__).parent))

from mochi_dataset import MOCHIDataset, parse_answer, create_labeled_grid
from mochi_naive_eval import EvalResult, EvalSummary


class MOCHIGridEvaluator:
    """
    Evaluator for MOCHI benchmark using single image grid approach.
    
    This evaluator merges multiple images into a labeled 2x2 grid,
    testing the model's ability to understand spatial layouts.
    """
    
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 64,
        grid_size: int = 512,
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            lora_path: Optional path to LoRA weights
            device: Device to run on
            max_new_tokens: Max tokens to generate
            grid_size: Size of each cell in the grid
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.grid_size = grid_size
        
        print(f"Loading model: {model_path}")
        
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
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess single image for InternVL."""
        processed = self.dynamic_preprocess(image, image_size=448, max_num=12)
        pixel_values = torch.stack([self.transform(p) for p in processed])
        return pixel_values.to(self.model.device, dtype=torch.bfloat16)
    
    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Generate response for a single image.
        
        Args:
            image: PIL image (the grid)
            prompt: Text prompt
        
        Returns:
            Model's response text
        """
        if self.model_type == "internvl":
            pixel_values = self._preprocess_image(image)
            full_prompt = f"<image>\n{prompt}"
            
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_prompt,
                    dict(max_new_tokens=self.max_new_tokens, do_sample=False),
                )
            return response
        
        else:
            inputs = self.processor(
                images=image,
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
        grid_image = trial["images"]  # Already a grid in single_image mode
        prompt = trial["prompt"]
        
        response = self.generate(grid_image, prompt)
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
        """Evaluate on entire MOCHI dataset."""
        results = []
        
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            trial = dataset[idx]
            result = self.evaluate_trial(trial)
            results.append(result)
        
        summary = self._compute_summary(results)
        
        if output_path:
            self._save_results(results, summary, output_path)
        
        return results, summary
    
    def _compute_summary(self, results: list[EvalResult]) -> EvalSummary:
        """Compute summary statistics."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        
        results_3obj = [r for r in results if r.n_objects == 3]
        results_4obj = [r for r in results if r.n_objects == 4]
        
        acc_3obj = sum(1 for r in results_3obj if r.correct) / len(results_3obj) if results_3obj else 0
        acc_4obj = sum(1 for r in results_4obj if r.correct) / len(results_4obj) if results_4obj else 0
        
        by_condition = {}
        for cond in set(r.condition for r in results):
            cond_results = [r for r in results if r.condition == cond]
            by_condition[cond] = {
                "total": len(cond_results),
                "correct": sum(1 for r in cond_results if r.correct),
                "accuracy": sum(1 for r in cond_results if r.correct) / len(cond_results),
            }
        
        by_dataset = {}
        for ds in set(r.dataset for r in results):
            ds_results = [r for r in results if r.dataset == ds]
            by_dataset[ds] = {
                "total": len(ds_results),
                "correct": sum(1 for r in ds_results if r.correct),
                "accuracy": sum(1 for r in ds_results if r.correct) / len(ds_results),
            }
        
        easy = [r for r in results if r.human_accuracy > 0.8]
        hard = [r for r in results if r.human_accuracy < 0.5]
        
        easy_acc = sum(1 for r in easy if r.correct) / len(easy) if easy else 0
        hard_acc = sum(1 for r in hard if r.correct) / len(hard) if hard else 0
        human_alignment = easy_acc - hard_acc
        
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
                "mode": "single_image",
                "grid_size": self.grid_size,
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
    parser = argparse.ArgumentParser(description="MOCHI Grid Evaluator")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA weights")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--subset", default=None, help="Filter by dataset")
    parser.add_argument("--condition", default=None, help="Filter by condition")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=512, help="Grid cell size")
    
    args = parser.parse_args()
    
    # Load dataset in single_image mode
    dataset = MOCHIDataset(
        mode="single_image",
        max_samples=args.max_samples,
        subset=args.subset,
        condition=args.condition,
    )
    
    evaluator = MOCHIGridEvaluator(
        model_path=args.model,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        grid_size=args.grid_size,
    )
    
    results, summary = evaluator.evaluate(dataset, args.output)
    
    print(f"\nEvaluation complete!")
    print(f"Overall accuracy: {summary.accuracy:.2%}")


if __name__ == "__main__":
    main()
