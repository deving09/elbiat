"""
Build training datasets from convos with feedback.

Usage:
    python scripts/build_training_dataset.py --type feedback_prediction --name feedback_v1 --test-ratio 0.1 --val-ratio 0.1
    python scripts/build_training_dataset.py --type answer_refinement --name refined_v1 --refine-model claude-3-sonnet
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import random
import json
from datetime import datetime
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models import Convo, Image
from app.db import DATABASE_URL




def get_convos_with_feedback(session: Session, min_feedback_len: int = 5):
    """Get all convos that have meaningful feedback."""
    query = (
        select(Convo)
        .where(Convo.feedback.isnot(None))
        .where(func.length(Convo.feedback) >= min_feedback_len)
    )
    return session.execute(query).scalars().all()



def parse_convo(convo) -> dict:
    """Parse convo JSON into prompt/response."""
    conversations = convo.conversations or []
    
    prompt = ""
    response = ""
    
    for turn in conversations:
        if turn.get("from") == "human":
            prompt = turn.get("value", "").replace("<image>\n", "").strip()
        elif turn.get("from") == "gpt":
            response = turn.get("value", "").strip()
    
    return {
        "prompt": prompt,
        "response": response,
    }



def split_data(items: list, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    """Split items into train/val/test."""
    random.seed(seed)
    items = items.copy()
    random.shuffle(items)
    
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    return {
        "train": items[:-(n_test + n_val)] if (n_test + n_val) > 0 else items,
        "val": items[-(n_test + n_val):-n_test] if n_test > 0 else items[-(n_test + n_val):],
        "test": items[-n_test:] if n_test > 0 else [],
    }

# ============== LLM Refinement ==============

class AnswerRefiner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print("Model loaded.")
    
    def refine(self, prompt: str, original_answer: str, feedback: str) -> str:
        """Generate refined answer based on feedback."""
        
        system = """You are helping improve VLM training data. Given an original question, 
the model's answer, and human feedback about what was wrong, generate an improved answer 
that addresses the feedback while maintaining accuracy. Output only the improved answer, nothing else."""
        
        user_prompt = f"""Original question: {prompt}

Model's answer: {original_answer}

Human feedback: {feedback}

Improved answer:"""
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Deterministic if temperature is 0
        do_sample = self.temperature > 0
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()



# ============== Dataset Builders ==============

def build_feedback_prediction(
    session: Session,
    output_dir: Path,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    min_feedback_len: int = 5,
    seed: int = 42,
):
    """
    Build dataset for feedback prediction.
    Input: image + question + model answer
    Output: feedback
    """
    convos = get_convos_with_feedback(session, min_feedback_len)
    print(f"Found {len(convos)} convos with feedback (>= {min_feedback_len} chars)")
    
    if not convos:
        print("No convos found!")
        return
    
    splits = split_data(convos, 1 - test_ratio - val_ratio, val_ratio, test_ratio, seed)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for split_name, split_convos in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        count = 0
        
        with open(output_file, "w") as f:
            for convo in split_convos:
                parsed = parse_convo(convo)
                
                if not parsed["prompt"] or not parsed["response"]:
                    continue
                
                image = session.get(Image, convo.image_id)
                
                record = {
                    "id": convo.id,
                    "image": image.image_path if image else None,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nQuestion: {parsed['prompt']}\n\nModel Answer: {parsed['response']}\n\nWhat feedback would you give about this answer?"
                        },
                        {
                            "from": "gpt",
                            "value": convo.feedback
                        }
                    ]
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        
        stats[split_name] = count
        print(f"  {split_name}: {count} examples -> {output_file}")
    
    # Write metadata
    meta = {
        "task": "feedback_prediction",
        "total": sum(stats.values()),
        "splits": stats,
        "min_feedback_len": min_feedback_len,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nTotal: {sum(stats.values())} examples")


def build_answer_refinement(
    session: Session,
    output_dir: Path,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    min_feedback_len: int = 10,
    seed: int = 42,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "auto",
):
    """
    Build dataset with LLM-refined answers.
    Uses feedback to generate improved answers for SFT.
    """
    convos = get_convos_with_feedback(session, min_feedback_len)
    print(f"Found {len(convos)} convos with feedback (>= {min_feedback_len} chars)")
    
    if not convos:
        print("No convos found!")
        return
    
    # Load refiner model
    refiner = AnswerRefiner(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    
    splits = split_data(convos, 1 - test_ratio - val_ratio, val_ratio, test_ratio, seed)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for split_name, split_convos in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        count = 0
        errors = 0
        
        with open(output_file, "w") as f:
            for i, convo in enumerate(split_convos):
                parsed = parse_convo(convo)
                
                if not parsed["prompt"] or not parsed["response"]:
                    continue
                
                print(f"  [{split_name}] {i+1}/{len(split_convos)}: refining convo {convo.id}...", end=" ")
                
                try:
                    refined = refiner.refine(
                        parsed["prompt"],
                        parsed["response"],
                        convo.feedback,
                    )
                    print(f"OK ({len(refined)} chars)")
                except Exception as e:
                    print(f"ERROR: {e}")
                    errors += 1
                    continue
                
                image = session.get(Image, convo.image_id)
                
                record = {
                    "id": convo.id,
                    "image": image.image_path if image else None,
                    "original_response": parsed["response"],
                    "feedback": convo.feedback,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{parsed['prompt']}"
                        },
                        {
                            "from": "gpt",
                            "value": refined
                        }
                    ]
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        
        stats[split_name] = {"count": count, "errors": errors}
        print(f"  {split_name}: {count} examples ({errors} errors) -> {output_file}")
    
    # Write metadata
    meta = {
        "task": "answer_refinement",
        "total": sum(s["count"] for s in stats.values()),
        "splits": stats,
        "min_feedback_len": min_feedback_len,
        "refiner_model": model_name,
        "temperature": temperature,
        "top_p": top_p,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nTotal: {sum(s['count'] for s in stats.values())} examples")

# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(description="Build training datasets from convos")
    subparsers = parser.add_subparsers(dest="task", required=True)
    
    # Feedback prediction
    fp_parser = subparsers.add_parser("feedback_prediction", help="Build feedback prediction dataset")
    fp_parser.add_argument("--output", "-o", required=True, help="Output directory")
    fp_parser.add_argument("--test-ratio", type=float, default=0.1)
    fp_parser.add_argument("--val-ratio", type=float, default=0.1)
    fp_parser.add_argument("--min-feedback-len", type=int, default=10)
    fp_parser.add_argument("--seed", type=int, default=42)
    
    # Answer refinement
    ar_parser = subparsers.add_parser("answer_refinement", help="Build answer refinement dataset")
    ar_parser.add_argument("--output", "-o", required=True, help="Output directory")
    ar_parser.add_argument("--test-ratio", type=float, default=0.1)
    ar_parser.add_argument("--val-ratio", type=float, default=0.1)
    ar_parser.add_argument("--min-feedback-len", type=int, default=10)
    ar_parser.add_argument("--seed", type=int, default=42)
    ar_parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="LLM for refinement")
    ar_parser.add_argument("--temperature", type=float, default=0.7, help="0 = deterministic, higher = more random")
    ar_parser.add_argument("--top-p", type=float, default=0.9)
    ar_parser.add_argument("--device", default="auto", help="Device for model")
    
    args = parser.parse_args()
    
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
        if args.task == "feedback_prediction":
            build_feedback_prediction(
                session,
                Path(args.output),
                test_ratio=args.test_ratio,
                val_ratio=args.val_ratio,
                min_feedback_len=args.min_feedback_len,
                seed=args.seed,
            )
        elif args.task == "answer_refinement":
            build_answer_refinement(
                session,
                Path(args.output),
                test_ratio=args.test_ratio,
                val_ratio=args.val_ratio,
                min_feedback_len=args.min_feedback_len,
                seed=args.seed,
                model_name=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )


if __name__ == "__main__":
    main()