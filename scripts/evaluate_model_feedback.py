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

from bert_score import score as bert_score
from transformers import AutoModelForCausalLM


# Add this constant after imports
FEEDBACK_PREDICTION_PROMPT = """Look at the image and analyze the following interaction.

Question: {question}

Model's Answer: {model_answer}

Based on the image, what feedback would you give about this answer? 
Point out any errors, missing information, or corrections needed."""



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



class LLMJudge:
    """Use an LLM to judge feedback quality."""
    
    JUDGE_PROMPT = """You are evaluating whether predicted feedback captures the same key points as reference feedback.

Question: {question}
Model's original answer: {model_answer}

Reference feedback (human): {reference}

Predicted feedback (model): {predicted}

Rate the predicted feedback:
1. Correctness: Does it identify the same errors? (0-1)
2. Completeness: Does it cover all key points? (0-1)
3. Accuracy: Is the predicted feedback factually correct? (0-1)

Respond in JSON only:
{{"correctness": <0-1>, "completeness": <0-1>, "accuracy": <0-1>, "overall": <0-1>}}"""


    IMPROVED_JUDGE_PROMPT = """You are comparing two pieces of feedback about a model's answer to a chart question.

REFERENCE (human-written, treat as ground truth):
"{reference}"

PREDICTED (model-generated, being evaluated):  
"{predicted}"

Step 1: What are the key claims in the REFERENCE?
Step 2: For each key claim, is it present in PREDICTED? (make sure key facts are exact but allow for different wording)
Step 3: Does PREDICTED contain any claims that contradict REFERENCE?

Score:
- 1.0 = All key claims present, no contradictions
- 0.75 = Most key claims present, no contradictions  
- 0.5 = Some key claims present OR minor contradictions
- 0.25 = Few key claims present OR major contradictions
- 0.0 = Missing most claims OR completely wrong

Respond with your reasoning, then JSON: {{"score": <0-1>, "matched_claims": [...], "missing_claims": [...], "contradictions": [...]}}"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading judge model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    def judge(self, question: str, model_answer: str, reference: str, predicted: str) -> dict:
        prompt = self.IMPROVED_JUDGE_PROMPT.format(
            question=question,
            model_answer=model_answer,
            reference=reference,
            predicted=predicted,
        )
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

       
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"score": 0}
    





class Evaluator:
    
    def __init__(
        self,
        model_path: str,
        root_dir: str = "/home/ubuntu/workspace/elbiat",
        use_semantic: bool = True,
        use_llm_judge: bool = False,
        judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_path: str = None, 
        base_model: str = "OpenGVLab/InternVL2_5-2B",
    ):
        self.root_dir = Path(root_dir)

        # Determine if model_path is a LoRA checkpoint or full model
        is_lora = lora_path is not None or (
        Path(model_path).exists() and 
        (Path(model_path) / "adapter_config.json").exists()
        )

        if is_lora:
            # Load base model first
            actual_lora_path = lora_path or model_path
            print(f"Loading base model: {base_model}")
            print(f"Loading LoRA weights: {actual_lora_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True, use_fast=False
            )
            self.model = InternVLChatModel.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            
            # Load and merge LoRA
            from peft import PeftModel
            self.model.language_model = PeftModel.from_pretrained(
                self.model.language_model,
                actual_lora_path,
            )
            self.model.language_model = self.model.language_model.merge_and_unload()
            print("Merged LoRA weights")

        else:
        
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
            )
            
        self.model = self.model.eval()
        self.transform = build_transform(is_train=False, input_size=448)
        
        # Semantic similarity model
        if use_semantic:
            print("Loading semantic similarity model...")
            self.sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.sem_model = None
        
        # LLM judge
        self.judge = LLMJudge(judge_model) if use_llm_judge else None
    
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
    
    def compute_metrics(self, prediction: str, reference: str, use_bert: bool = False) -> dict:
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
        

        metrics = {
            "exact_match": exact,
            "contains_match": contains,
            "numeric_precision": num_p,
            "numeric_recall": num_r,
            "numeric_f1": num_f1,
            "term_precision": term_p,
            "term_recall": term_r,
            "term_f1": t_f1,
            "semantic_similarity": sem_sim,
        }

        # BERTScore (optional, slower)
        if use_bert:
            P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
            metrics["bert_score"] = float(F1[0])
        
        return metrics

    
    def parse_sample(self, sample: dict) -> dict:
        """Parse feedback data format."""
        question = ""
        model_answer = ""
        human_feedback = ""
        
        for turn in sample.get("conversations", []):
            if turn["from"] == "human":
                text = turn["value"].replace("<image>\n", "")
                
                if "Question:" in text:
                    q_part = text.split("Question:")[-1]
                    question = q_part.split("\n")[0].strip()
                
                if "Model Answer:" in text:
                    a_part = text.split("Model Answer:")[-1]
                    if "What feedback" in a_part:
                        model_answer = a_part.split("What feedback")[0].strip()
                    else:
                        model_answer = a_part.split("\n")[0].strip()
                        
            elif turn["from"] == "gpt":
                human_feedback = turn["value"].strip()
        
        return {
            "id": sample.get("id"),
            "image": sample["image"],
            "question": question,
            "model_answer": model_answer,
            "human_feedback": human_feedback,
        }


    def evaluate_feedback_prediction(self, test_data: list[dict], use_bert: bool = True) -> dict:
        """Evaluate feedback prediction task."""
        results = []
        
        for sample in tqdm(test_data, desc="Evaluating feedback prediction"):
            parsed = self.parse_sample(sample)
            
            if not parsed["question"] or not parsed["human_feedback"]:
                print(f"Skipping sample {parsed['id']}: missing data")
                continue
            
            # Build prompt for feedback prediction
            prompt = FEEDBACK_PREDICTION_PROMPT.format(
                question=parsed["question"],
                model_answer=parsed["model_answer"],
            )
            
            # Generate prediction
            prediction = self.generate(parsed["image"], prompt)
            reference = parsed["human_feedback"]
            
            # Compute metrics
            metrics = self.compute_metrics(prediction, reference, use_bert=use_bert)
            
            # LLM judge
            if self.judge:
                judge_result = self.judge.judge(
                    parsed["question"],
                    parsed["model_answer"],
                    reference,
                    prediction,
                )
                metrics["llm_judge"] = judge_result
            
            print(f"\n--- Sample {parsed['id']} ---")
            print(f"Q: {parsed['question']}")
            print(f"Model answer: {parsed['model_answer'][:100]}...")
            print(f"Human feedback: {reference[:100]}...")
            print(f"Predicted: {prediction[:100]}...")
            print(f"Semantic sim: {metrics['semantic_similarity']:.2%}")
            
            results.append({
                "id": parsed["id"],
                "image": parsed["image"],
                "question": parsed["question"],
                "model_answer": parsed["model_answer"],
                "human_feedback": reference,
                "predicted_feedback": prediction,
                "metrics": metrics,
            })
        
        # Aggregate
        if not results:
            return {"error": "No valid samples"}
        
        aggregate = {
            "num_samples": len(results),
            "semantic_similarity": np.mean([r["metrics"]["semantic_similarity"] for r in results]),
            "numeric_f1": np.mean([r["metrics"]["numeric_f1"] for r in results]),
            "term_f1": np.mean([r["metrics"]["term_f1"] for r in results]),
        }
        
        if use_bert:
            aggregate["bert_score"] = np.mean([r["metrics"]["bert_score"] for r in results])
        
        if self.judge:
            judge_scores = [r["metrics"]["llm_judge"] for r in results]
            aggregate["llm_judge"] = {
                "llm_score": np.mean([s.get("score", 0) for s in judge_scores]), 
                #"correctness": np.mean([s.get("correctness", 0) for s in judge_scores]),
                #"completeness": np.mean([s.get("completeness", 0) for s in judge_scores]),
                #"accuracy": np.mean([s.get("accuracy", 0) for s in judge_scores]),
                #"overall": np.mean([s.get("overall", 0) for s in judge_scores]),
            }
        
        return {
            "aggregate": aggregate,
            "per_sample": results,
        }

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
    parser.add_argument("--task", choices=["correction", "feedback_prediction"], default="feedback_prediction")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-bert", action="store_true", help="Skip BERTScore (faster)")
    parser.add_argument("--base-model", default="OpenGVLab/InternVL2_5-2B")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA weights (if separate from --model)")

    
    args = parser.parse_args()
    
    # Load data
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    evaluator = Evaluator(
        args.model,
        args.root_dir,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
        lora_path=args.lora_path,
        base_model=args.base_model,
    )
    
    if args.task == "feedback_prediction":
        results = evaluator.evaluate_feedback_prediction(test_data, use_bert=not args.no_bert)
    else:
        results = evaluator.evaluate_dataset(test_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({args.task})")
    print("=" * 60)
    agg = results["aggregate"]
    print(f"Samples: {agg['num_samples']}")
    
    for key, val in agg.items():
        if key == "num_samples":
            continue
        if isinstance(val, dict):
            print(f"\n{key}:")
            for k, v in val.items():
                print(f"  {k}: {v:.2%}")
        else:
            print(f"{key}: {val:.2%}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
