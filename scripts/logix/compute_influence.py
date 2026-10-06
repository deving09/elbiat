# scripts/logix/compute_influence.py
"""
LogIX-based influence function computation for InternVL LoRA models.

Usage:
    # Step 1: Extract training logs (run once per checkpoint)
    python scripts/logix/compute_influence.py extract-logs \
        --model checkpoints/feedback_v1/ \
        --data feedback_data/feedback_v1/train.jsonl \
        --output logix_logs/feedback_v1/

    # Step 2: Compute influence for a benchmark
    python scripts/logix/compute_influence.py compute \
        --model checkpoints/feedback_v1/ \
        --logs logix_logs/feedback_v1/ \
        --benchmark chartqa \
        --output logix_results/chartqa_attribution.json

    # Or run both steps together
    python scripts/logix/compute_influence.py run-all \
        --model checkpoints/feedback_v1/ \
        --data feedback_data/feedback_v1/train.jsonl \
        --benchmark chartqa \
        --output logix_results/chartqa_attribution.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import tempfile
import base64

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from tqdm import tqdm
from PIL import Image


import collections
from logix.utils import nested_dict

torch.serialization.add_safe_globals([collections.defaultdict, nested_dict])
# Add project paths
#sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/ubuntu/workspace/elbiat/external/InternVL/internvl_chat')
# Add paths
#sys.path.insert(0, '/home/ubuntu/workspace/elbiat/external/InternVL/internvl_chat')

#from internvl.model.internvl_chat import InternVLChatModel
import logix
from logix.config import LoRAConfig

import logix.utils
_original_to_numpy = logix.utils.to_numpy

def _patched_to_numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return _original_to_numpy(tensor)

logix.utils.to_numpy = _patched_to_numpy

# Also patch it where it's already imported
logix.logging.log_saver.to_numpy = _patched_to_numpy

"""
def load_model(model_path: str, device: str = "cuda"):
    from transformers import AutoTokenizer
    from internvl.model.internvl_chat import InternVLChatModel
    
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    # Enable gradients for LoRA layers
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    # Set img_context_token_id
    if hasattr(tokenizer, 'added_tokens_encoder') and '<IMG_CONTEXT>' in tokenizer.added_tokens_encoder:
        model.img_context_token_id = tokenizer.added_tokens_encoder['<IMG_CONTEXT>']
    
    return model, tokenizer
"""

def load_model(model_path: str, device: str = "cuda"):
    """Load InternVL model and merge LoRA into base weights."""
    from transformers import AutoTokenizer
    from internvl.model.internvl_chat import InternVLChatModel
    
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    # Merge PEFT LoRA into base model so LogIX can add its own compression LoRA
    if hasattr(model.language_model, 'merge_and_unload'):
        print("Merging PEFT LoRA into base model...")
        model.language_model = model.language_model.merge_and_unload()
    
    model.train()
    # After merge, all params are in base model - enable grads on all
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Set img_context_token_id
    if hasattr(tokenizer, 'added_tokens_encoder') and '<IMG_CONTEXT>' in tokenizer.added_tokens_encoder:
        model.img_context_token_id = tokenizer.added_tokens_encoder['<IMG_CONTEXT>']
    
    return model, tokenizer



def load_training_examples(data_path: str, images_dir: str = "/home/ubuntu/workspace/elbiat/images") -> List[Dict]:
    """Load training examples from JSONL."""
    examples = []
    base_dir = Path(data_path).parent
    
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            
            # Parse conversation format
            conversations = item.get("conversations", [])
            question = ""
            answer = ""
            for turn in conversations:
                if turn.get("from") == "human":
                    question = turn.get("value", "").replace("<image>\n", "").strip()
                elif turn.get("from") == "gpt":
                    answer = turn.get("value", "").strip()
            
            image_path = item.get("image", "")
            image_name = Path(image_path).name
            image_path = str(Path(images_dir)/ image_name)

            
            examples.append({
                "id": item.get("id", len(examples)),
                "image": image_path,
                "question": question,
                "answer": answer,
            })
    
    return examples


def load_benchmark_examples(benchmark: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test examples from a benchmark using pandas directly."""
    import pandas as pd
    
    benchmark_map = {
        "chartqa": "ChartQA_TEST",
        "mochi_grid": "MOCHI_Grid",
        "mochi_naive": "MOCHI_Naive",
        "blink": "BLINK",
        "cvbench": "CV-Bench",
    }
    
    dataset_name = benchmark_map.get(benchmark.lower(), benchmark)
    lmu_root = os.environ.get("LMUData", "/home/ubuntu/LMUData")
    data_path = os.path.join(lmu_root, f"{dataset_name}.tsv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")
    
    df = pd.read_csv(data_path, sep='\t')
    
    examples = []
    temp_dir = tempfile.mkdtemp()
    
    for idx, row in df.iterrows():
        if max_samples and idx >= max_samples:
            break
        
        # Handle image - could be path or base64
        image_data = row.get("image", "")
        image_path = None
        
        if pd.notna(image_data):
            if isinstance(image_data, str) and len(image_data) > 500:
                # Likely base64
                try:
                    img_bytes = base64.b64decode(image_data)
                    temp_path = os.path.join(temp_dir, f"{benchmark}_{idx}.png")
                    with open(temp_path, 'wb') as f:
                        f.write(img_bytes)
                    image_path = temp_path
                except:
                    continue
            elif os.path.exists(str(image_data)):
                image_path = str(image_data)
        
        if not image_path:
            continue
        
        examples.append({
            "id": f"{benchmark}_{idx}",
            "image": image_path,
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        })
    
    return examples


def prepare_batch(
    model,
    tokenizer,
    example: Dict,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Prepare a single example for forward pass."""
    from internvl.train.dataset import build_transform, dynamic_preprocess
    from internvl.conversation import get_conv_template
    
    transform = build_transform(is_train=False, input_size=448)
    
    # Load and preprocess image
    image = Image.open(example["image"]).convert("RGB")
    patches = dynamic_preprocess(image, image_size=448, max_num=6)
    pixel_values = torch.stack([transform(p) for p in patches])
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    
    num_patches = pixel_values.shape[0]
    num_image_tokens = num_patches * model.num_image_token
    
    # Build prompt with image tokens
    image_token_str = '<IMG_CONTEXT>' * num_image_tokens
    question = example["question"]
    answer = example["answer"]
    
    template = get_conv_template(model.template)
    template.append_message(template.roles[0], f"<image>\n{question}")
    template.append_message(template.roles[1], answer)
    full_prompt = template.get_prompt()
    full_prompt = full_prompt.replace('<image>', image_token_str, 1)
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Find answer start for labels
    prompt_only = full_prompt.split(answer)[0]
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=True)
    prompt_len = prompt_tokens.input_ids.shape[1]
    
    # Create labels with -100 for prompt
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    
    # Get visual embeddings
    vit_embeds = model.extract_feature(pixel_values)
    
    return {
        "pixel_values": pixel_values,
        "vit_embeds": vit_embeds,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def forward_with_loss(model, batch: Dict) -> torch.Tensor:
    """Run forward pass and return loss."""
    vit_embeds = batch["vit_embeds"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    # Get text embeddings
    text_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    
    # Find and replace image token positions
    img_context_token_id = model.img_context_token_id
    img_positions = (input_ids == img_context_token_id)
    
    # Flatten vit_embeds to match image token count
    num_img_tokens = img_positions.sum().item()
    vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])[:num_img_tokens]
    
    # Replace image tokens with visual embeddings
    text_embeds[img_positions] = vit_flat.to(text_embeds.dtype)
    
    # Forward through language model
    outputs = model.language_model(
        inputs_embeds=text_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    
    return outputs.loss


def extract_training_logs(
    model_path: str,
    data_path: str,
    output_dir: str,
    lora_rank: int = 64,
    device: str = "cuda",
):
    """Extract gradient logs for all training examples."""
    model, tokenizer = load_model(model_path, device)
    examples = load_training_examples(data_path)
    
    print(f"Loaded {len(examples)} training examples")

    # Use output_dir as project name so logs go to logix_logs/<output_dir>/
    project_name = Path(output_dir).name  # e.g., "feedback_v1"
    
    # Initialize LogIX with project name matching output
    run = logix.init(project=project_name, config="configs/logix_config.yaml")
    
    
    # Watch LoRA layers
    #run.watch(model, name_filter=["lora_"], type_filter=[nn.Linear])
    run.watch(model, name_filter=["wqkv", "wo", "w1", "w2", "w3"], type_filter=[nn.Linear])
    
    # Add LoRA compression
    lora_config = LoRAConfig(init="random", rank=lora_rank)
    run.add_lora(lora_config=lora_config)

    # Ensure all model params are bfloat16
    model.to(torch.bfloat16) 
    
    # Setup gradient logging
    run.setup({"grad": ["log", "covariance"]})
    run.save(True)

    #for name, module in run.model.named_modules():
    #    if 'lora' in name.lower():
    #        print(f'{name}: {type(module).__name__}')
    #1/0
    
    # Extract logs for each example
    for example in tqdm(examples, desc="Extracting training logs"):
        try:
            batch = prepare_batch(model, tokenizer, example, device)
            
            with run(data_id=[str(example["id"])], mask=batch["attention_mask"]):
                model.zero_grad()
                loss = forward_with_loss(model, batch)
                loss.backward()
        except Exception as e:
            import traceback
            print(f"Error on example {example['id']}: {traceback.format_exc()}")
            #print(f"Error on example {example['id']}: {e}")
            continue
    
    # Finalize and save
    run.finalize()
    print(f"Training logs saved to {output_dir}")


def compute_influence(
    model_path: str,
    logs_dir: str,
    benchmark: str,
    output_path: str,
    max_test_samples: int = 500,
    lora_rank: int = 64,
    device: str = "cuda",
):
    """Compute influence scores for benchmark examples."""
    model, tokenizer = load_model(model_path, device)
    test_examples = load_benchmark_examples(benchmark, max_samples=max_test_samples)
    
    print(f"Loaded {len(test_examples)} test examples for {benchmark}")
    
    # Initialize LogIX and load training logs
    # Use output_dir as project name so logs go to logix_logs/<output_dir>/
    project_name = Path(logs_dir).name  # e.g., "feedback_v1"
    run = logix.init(project=project_name, config="configs/logix_config.yaml")
    run.watch(model, name_filter=["wqkv", "wo", "w1", "w2", "w3"], type_filter=[nn.Linear])
    #run.watch(model, name_filter=["lora_"], type_filter=[nn.Linear])
    
    lora_config = LoRAConfig(init="random", rank=lora_rank)
    run.add_lora(lora_config=lora_config)

    # Add this - need to setup logging for test examples too
    run.setup({"grad": ["log"]})
    
    # Ensure all model params are bfloat16
    model.to(torch.bfloat16) 

    
    # Load training logs
    run.initialize_from_log(logs_dir)
    log_loader = run.build_log_dataloader()
    
    # Aggregate influence across all test examples
    all_influences = {}
    
    for test_example in tqdm(test_examples, desc="Computing influence"):
        try:
            batch = prepare_batch(model, tokenizer, test_example, device)
            
            with run(data_id=[str(test_example["id"])], mask=batch["attention_mask"]):
                model.zero_grad()
                loss = forward_with_loss(model, batch)
                loss.backward()
            
            test_log = run.get_log()
            
            # Compute influence against all training examples
            influence_result = run.compute_influence_all(
                src_log=test_log,
                loader=log_loader,
                mode="dot",
                precondition=False,
            )

            # Accumulate scores
            train_ids = influence_result.get("tgt_ids", [])  # Changed from "data_id"
            scores = influence_result.get("influence", torch.zeros(1, len(train_ids)))

            if isinstance(scores, torch.Tensor):
                scores = scores.squeeze(0).cpu().numpy()  # Remove batch dim [1, 104] -> [104]

            for tid, score in zip(train_ids, scores):
                if tid not in all_influences:
                    all_influences[tid] = []
                all_influences[tid].append(float(score))
        
        except Exception as e:
            import traceback
            print(f"Error on example {test_example['id']}: {traceback.format_exc()}")
            print(f"Error on test example {test_example['id']}: {e}")
            continue
    
    # Average influence across test examples
    train_ids = list(all_influences.keys())
    influence_scores = [np.mean(all_influences[tid]) for tid in train_ids]
    
    # Compute z-scores and ranks
    scores_arr = np.array(influence_scores)
    z_scores = stats.zscore(scores_arr) if len(scores_arr) > 1 else np.zeros_like(scores_arr)
    ranks = stats.rankdata(-scores_arr, method='ordinal')
    
    # Build results
    results = {
        "train_ids": [int(tid) if tid.isdigit() else tid for tid in train_ids],
        "influence_scores": influence_scores,
        "stats": {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "n_positive": int(np.sum(scores_arr > 0)),
            "n_negative": int(np.sum(scores_arr < 0)),
        },
        "ranked": [
            {"id": train_ids[i], "score": float(influence_scores[i]), "rank": int(ranks[i])}
            for i in np.argsort(-scores_arr)
        ],
        "metadata": {
            "method": "logix",
            "benchmark": benchmark,
            "n_test_examples": len(test_examples),
            "lora_rank": lora_rank,
            "computed_at": datetime.utcnow().isoformat(),
        }
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved influence scores to {output_path}")
    print(f"Stats: {results['stats']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="LogIX Influence Functions for InternVL")
    subparsers = parser.add_subparsers(dest="command")
    
    # extract-logs command
    p_extract = subparsers.add_parser("extract-logs", help="Extract training gradient logs")
    p_extract.add_argument("--model", required=True, help="Model checkpoint path")
    p_extract.add_argument("--data", required=True, help="Training data JSONL")
    p_extract.add_argument("--output", required=True, help="Output directory for logs")
    p_extract.add_argument("--lora-rank", type=int, default=64, help="LoRA rank for compression")
    
    # compute command
    p_compute = subparsers.add_parser("compute", help="Compute influence scores")
    p_compute.add_argument("--model", required=True, help="Model checkpoint path")
    p_compute.add_argument("--logs", required=True, help="Training logs directory")
    p_compute.add_argument("--benchmark", required=True, help="Benchmark name")
    p_compute.add_argument("--output", required=True, help="Output JSON path")
    p_compute.add_argument("--max-samples", type=int, default=500, help="Max test samples")
    p_compute.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    
    # run-all command
    p_all = subparsers.add_parser("run-all", help="Extract logs and compute influence")
    p_all.add_argument("--model", required=True, help="Model checkpoint path")
    p_all.add_argument("--data", required=True, help="Training data JSONL")
    p_all.add_argument("--benchmark", required=True, help="Benchmark name")
    p_all.add_argument("--output", required=True, help="Output JSON path")
    p_all.add_argument("--max-samples", type=int, default=500, help="Max test samples")
    p_all.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    
    args = parser.parse_args()
    
    if args.command == "extract-logs":
        extract_training_logs(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            lora_rank=args.lora_rank,
        )
    elif args.command == "compute":
        compute_influence(
            model_path=args.model,
            logs_dir=args.logs,
            benchmark=args.benchmark,
            output_path=args.output,
            max_test_samples=args.max_samples,
            lora_rank=args.lora_rank,
        )
    elif args.command == "run-all":
        logs_dir = f"logix_logs/{Path(args.model).name}"
        extract_training_logs(
            model_path=args.model,
            data_path=args.data,
            output_dir=logs_dir,
            lora_rank=args.lora_rank,
        )
        compute_influence(
            model_path=args.model,
            logs_dir=logs_dir,
            benchmark=args.benchmark,
            output_path=args.output,
            max_test_samples=args.max_samples,
            lora_rank=args.lora_rank,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()