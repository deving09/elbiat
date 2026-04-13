"""
CLI for TracIn data attribution.

Usage:
    # Step 1: Compute gradients for training data
    python run_attribution.py compute-gradients \
        --model checkpoints/dpo_v1/best \
        --data feedback_data/feedback_v1/train.jsonl \
        --output tracin_cache/ \
        --split train

    # Step 2: Compute gradients for each benchmark
    python run_attribution.py compute-gradients \
        --model checkpoints/dpo_v1/best \
        --benchmark chartqa \
        --output tracin_cache/ \
        --split test

    # Step 3: Compute attribution scores
    python run_attribution.py attribute \
        --train-grads tracin_cache/train_gradients.npz \
        --test-grads tracin_cache/test_chartqa_gradients.npz \
        --output tracin_results/chartqa_attribution.json

    # Or all benchmarks at once
    python run_attribution.py attribute-all \
        --train-grads tracin_cache/train_gradients.npz \
        --output tracin_results/all_attribution.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compute_gradients import GradientConfig, compute_gradients_for_dataset
from attribute import compute_influence_scores, compute_multi_benchmark_attribution

#from vlmeval.smp import load#, #LMUDataRoot


import os
import base64
from io import BytesIO
from PIL import Image
import tempfile


dummy_point = {"id": 17, 
"image": "images/21_sajk.png", 
"conversations": [{"from": "human", "value": "<image>\nQuestion: What year has the highest peak\n\nModel Answer: The highest peak in sales volume is in 2008, with 22.12 million units sold worldwide.\n\nWhat feedback would you give about this answer?"}, {"from": "gpt", "value": "the actual peak is in 2009 with 22.73 million sales"}]
}

dummy_example = {
    "id": dummy_point["id"],
    "image": dummy_point["image"],
    "question": dummy_point["conversations"][0]["value"].replace("<image>\n", ""),
    "answer": dummy_point["conversations"][1]["value"]
}

def load_training_examples(data_path: str, max_samples: int = None) -> list[dict]:
    """Load training examples from JSONL."""
    examples = []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            item = json.loads(line)
            # Extract from conversation format
            convos = item["conversations"]
            question = convos[0]["value"].replace("<image>\n", "")
            answer = convos[1]["value"]
            
            examples.append({
                "id": item["id"],
                "image": item["image"],
                "question": question,
                "answer": answer,
            })
    return examples





def load_benchmark_examples(benchmark: str, max_samples: int = None) -> list[dict]:
    """Load test examples from a benchmark."""
    """
    from vlmeval.smp import load, LMUDataRoot
    import os
    import base64
    from io import BytesIO
    from PIL import Image
    import tempfile
    """

    import pandas as pd
    import base64
    from io import BytesIO
    from PIL import Image
    import tempfile
    
    benchmark_map = {
        "chartqa": "ChartQA_TEST",
        "mochi_grid": "MOCHI_Grid",
        "mochi_naive": "MOCHI_Naive", 
        "blink": "BLINK",
        "cvbench": "CV-Bench",
    }
    
    dataset_name = benchmark_map.get(benchmark.lower(), benchmark)
    lmu_root = os.path.expanduser("~/LMUData")
    data_path = os.path.join(lmu_root, f"{dataset_name}.tsv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")
    
    data = pd.read_csv(data_path, sep='\t')

    """
    benchmark_map = {
        "chartqa": "ChartQA_TEST",
        "mochi_grid": "MOCHI_Grid",
        "mochi_naive": "MOCHI_Naive", 
        "blink": "BLINK",
        "cvbench": "CV-Bench",
    }
    
    dataset_name = benchmark_map.get(benchmark.lower(), benchmark)
    data_path = os.path.join(LMUDataRoot(), f"{dataset_name}.tsv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")
    
    data = load(data_path)
    """
    
    # Create temp dir for decoded images
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tracin_{benchmark}_"))
    print(f"Decoding images to {temp_dir}")
    
    examples = []
    for idx, row in data.iterrows():
        if max_samples and idx >= max_samples:
            break
        
        # Handle image - could be path or base64
        image_data = row.get("image", "")
        image_path = row.get("image_path", "")
        
        # If image_path exists and is a real file, use it
        if image_path and Path(image_path).exists():
            final_path = image_path
        elif image_data:
            # Decode base64
            try:
                img_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(img_bytes))
                final_path = temp_dir / f"{idx}.png"
                img.save(final_path)
            except Exception as e:
                print(f"Failed to decode image {idx}: {e}")
                continue
        else:
            continue
        
        examples.append({
            "id": f"{benchmark}_{idx}",
            "image": str(final_path),
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")),
        })
    
    print(f"Loaded {len(examples)} examples from {benchmark}")
    return examples


def cmd_compute_gradients(args):
    """Compute gradients for a dataset."""
    config = GradientConfig(
        model_path=args.model,
        output_dir=args.output,
        gradient_layers=args.layers,
    )
    
    if args.split == "train":
        examples = load_training_examples(args.data, args.max_samples)
    else:
        examples = load_benchmark_examples(args.benchmark, args.max_samples)
    
    print(f"Loaded {len(examples)} examples")
    
    compute_gradients_for_dataset(
        config=config,
        examples=examples,
        split=args.split,
        benchmark=args.benchmark if args.split == "test" else None,
    )


def cmd_attribute(args):
    """Compute attribution scores."""
    compute_influence_scores(
        train_gradients_path=args.train_grads,
        test_gradients_path=args.test_grads,
        output_path=args.output,
        normalize=not args.no_normalize,
        approach=args.approach
    )


def cmd_attribute_all(args):
    """Compute attribution for all benchmarks."""
    cache_dir = Path(args.train_grads).parent
    
    benchmark_paths = {}
    for benchmark in ["chartqa", "mochi_grid", "blink", "cvbench"]:
        path = cache_dir / f"test_{benchmark}_gradients.npz"
        if path.exists():
            benchmark_paths[benchmark] = str(path)
        else:
            print(f"Warning: {path} not found, skipping {benchmark}")
    
    if not benchmark_paths:
        print("No benchmark gradients found!")
        return
    
    compute_multi_benchmark_attribution(
        train_gradients_path=args.train_grads,
        benchmark_gradient_paths=benchmark_paths,
        output_path=args.output,
    )


def main():
    parser = argparse.ArgumentParser(description="TracIn Data Attribution")
    subparsers = parser.add_subparsers(dest="command")
    
    # compute-gradients
    p_grads = subparsers.add_parser("compute-gradients")
    p_grads.add_argument("--model", required=True, help="Model checkpoint path")
    p_grads.add_argument("--output", required=True, help="Output directory")
    p_grads.add_argument("--split", choices=["train", "test"], required=True)
    p_grads.add_argument("--data", help="Training data JSONL (for split=train)")
    p_grads.add_argument("--benchmark", help="Benchmark name (for split=test)")
    p_grads.add_argument("--max-samples", type=int, help="Limit test samples")
    p_grads.add_argument("--layers", default="lora", choices=["lora", "llm", "all"])
    
    # attribute
    p_attr = subparsers.add_parser("attribute")
    p_attr.add_argument("--train-grads", required=True)
    p_attr.add_argument("--test-grads", required=True)
    p_attr.add_argument("--output", required=True)
    p_attr.add_argument("--no-normalize", action="store_true")
    p_attr.add_argument("--approach", default="mean")
    
    # attribute-all
    p_all = subparsers.add_parser("attribute-all")
    p_all.add_argument("--train-grads", required=True)
    p_all.add_argument("--output", required=True)
    
    args = parser.parse_args()
    
    if args.command == "compute-gradients":
        cmd_compute_gradients(args)
    elif args.command == "attribute":
        cmd_attribute(args)
    elif args.command == "attribute-all":
        cmd_attribute_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
