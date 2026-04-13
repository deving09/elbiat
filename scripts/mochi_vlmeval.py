"""
MOCHI Dataset wrapper for VLMEvalKit integration.

This module provides VLMEvalKit-compatible dataset classes for MOCHI benchmark.
Place this in your elbiat directory and register in your custom datasets.

Two dataset variants:
- MOCHI_Naive: Multi-image evaluation (tests multi-image understanding)
- MOCHI_Grid: Single grid image (tests spatial layout understanding)
"""

import os
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset


# Index to letter mapping
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def create_labeled_grid(
    images: list,
    cell_size: int = 512,
    font_size: int = 48,
) -> Image.Image:
    """Create a 2x2 grid with labels A, B, C, D."""
    n_images = len(images)
    
    grid_width = cell_size * 2
    grid_height = cell_size * 2
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]
    draw = ImageDraw.Draw(grid)
    
    for i, (img, (x, y)) in enumerate(zip(images, positions)):
        img_size = cell_size - font_size - 20
        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        
        paste_x = x + (cell_size - img_size) // 2
        paste_y = y + font_size + 10
        grid.paste(img_resized, (paste_x, paste_y))
        
        label = INDEX_TO_LETTER[i]
        label_x = x + cell_size // 2
        label_y = y + 5
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font, anchor="mt")
    
    if n_images == 3:
        x, y = positions[3]
        center_x = x + cell_size // 2
        center_y = y + cell_size // 2
        draw.text((center_x, center_y), "(empty)", fill=(128, 128, 128), font=font, anchor="mm")
    
    return grid


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def parse_mochi_answer(response: str, n_objects: int = 4) -> Optional[str]:
    """Parse model response to extract answer letter."""
    response = response.strip().upper()
    valid_answers = ["A", "B", "C"] if n_objects == 3 else ["A", "B", "C", "D"]
    
    if response in valid_answers:
        return response
    
    for letter in valid_answers:
        patterns = [
            f"IMAGE {letter}", f"OPTION {letter}", f"ANSWER: {letter}",
            f"({letter})", f"[{letter}]", f"{letter}.", f"{letter})",
            f"THE ODD-ONE-OUT IS {letter}", f"ODD ONE OUT IS {letter}",
        ]
        for pattern in patterns:
            if pattern in response:
                return letter
    
    for letter in valid_answers:
        if f" {letter} " in f" {response} " or response.startswith(letter) or response.endswith(letter):
            return letter
    
    return None


class MOCHIBaseDataset:
    """
    Base class for MOCHI dataset integration with VLMEvalKit.
    """
    
    # Dataset metadata
    DATASET_URL = "https://huggingface.co/datasets/tzler/MOCHI"
    TYPE = "MCQ"  # Multiple choice question
    
    def __init__(self, dataset_name: str = "MOCHI"):
        self.dataset_name = dataset_name
        self._data = None
        self._hf_dataset = None
    
    def load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        if self._hf_dataset is None:
            print(f"Loading MOCHI dataset from HuggingFace...")
            self._hf_dataset = load_dataset("tzler/MOCHI", split="train")
        return self._hf_dataset
    
    def build_tsv(self, output_path: str):
        """Build TSV file for VLMEvalKit (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from TSV file."""
        return pd.read_csv(data_path, sep="\t")
    
    def build_prompt(self, line: dict) -> list:
        """Build prompt for a single sample (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def evaluate(self, eval_file: str, **judge_kwargs) -> dict:
        """
        Evaluate predictions.
        
        Args:
            eval_file: Path to prediction file (xlsx format from VLMEvalKit)
        
        Returns:
            Dictionary with evaluation metrics
        """
        df = pd.read_excel(eval_file)
        
        results = {
            "total": len(df),
            "correct": 0,
            "by_n_objects": {"3": {"total": 0, "correct": 0}, "4": {"total": 0, "correct": 0}},
            "by_condition": {},
        }
        
        for _, row in df.iterrows():
            n_objects = row.get("n_objects", 4)
            condition = row.get("category", "unknown")
            answer = row["answer"]
            prediction = row.get("prediction", "")
            
            # Parse prediction
            parsed = parse_mochi_answer(str(prediction), n_objects)
            correct = parsed == answer
            
            results["correct"] += int(correct)
            
            # By n_objects
            key = str(n_objects)
            results["by_n_objects"][key]["total"] += 1
            results["by_n_objects"][key]["correct"] += int(correct)
            
            # By condition
            if condition not in results["by_condition"]:
                results["by_condition"][condition] = {"total": 0, "correct": 0}
            results["by_condition"][condition]["total"] += 1
            results["by_condition"][condition]["correct"] += int(correct)
        
        # Compute accuracies
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
        
        for key in results["by_n_objects"]:
            stats = results["by_n_objects"][key]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        for cond in results["by_condition"]:
            stats = results["by_condition"][cond]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        return results


class MOCHI_Naive(MOCHIBaseDataset):
    """
    MOCHI dataset with naive multi-image approach.
    
    Each sample contains multiple separate images that are passed
    to the VLM model simultaneously.
    
    Note: This requires VLM models that support multiple image inputs.
    """
    
    MODALITY = "multi-image"
    
    def __init__(self):
        super().__init__("MOCHI_Naive")
    
    def build_tsv(self, output_path: str, image_dir: Optional[str] = None):
        """
        Build TSV file for VLMEvalKit.
        
        Note: Standard VLMEvalKit doesn't support multi-image well,
        so we store additional image paths in a separate column.
        """
        hf_data = self.load_hf_dataset()
        
        if image_dir is None:
            image_dir = tempfile.mkdtemp(prefix="mochi_naive_")
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for idx, trial in enumerate(hf_data):
            images = trial["images"]
            n_objects = trial["n_objects"]
            oddity_index = trial["oddity_index"]
            answer = INDEX_TO_LETTER[oddity_index]
            
            # Save all images
            image_paths = []
            for i, img in enumerate(images):
                img_path = image_dir / f"{trial['trial']}_{i}.png"
                img.save(img_path)
                image_paths.append(str(img_path))
            
            # Encode first image as base64 for VLMEvalKit
            img_b64 = image_to_base64(images[0])
            
            # Build prompt
            if n_objects == 3:
                prompt = (
                    "You are shown 3 images. Each shows an object from a different viewpoint. "
                    "Two show the same 3D object, one is different (the odd-one-out). "
                    "Which is the odd-one-out? Answer with just: A (Image 1), B (Image 2), or C (Image 3)."
                )
                options = {"A": "Image 1", "B": "Image 2", "C": "Image 3", "D": ""}
            else:
                prompt = (
                    "You are shown 4 images. Each shows an object from a different viewpoint. "
                    "Three show the same 3D object, one is different (the odd-one-out). "
                    "Which is the odd-one-out? Answer with: A (Image 1), B (Image 2), C (Image 3), or D (Image 4)."
                )
                options = {"A": "Image 1", "B": "Image 2", "C": "Image 3", "D": "Image 4"}
            
            rows.append({
                "index": idx,
                "image": img_b64,
                "question": prompt,
                "answer": answer,
                **options,
                "n_objects": n_objects,
                "category": trial["condition"],
                "dataset_source": trial["dataset"],
                "trial_id": trial["trial"],
                "human_accuracy": trial["human_avg"],
                # Store additional image paths for custom model wrapper
                "image_paths": "|".join(image_paths),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Built MOCHI_Naive TSV with {len(rows)} samples at {output_path}")
        return output_path
    
    def build_prompt(self, line) -> list:
        """
        Build multi-modal prompt for VLMEvalKit.
        
        Returns list of message dicts with multiple images.
        """
        if isinstance(line, int):
            line = self._data.iloc[line]
        
        # Get all image paths
        image_paths = line.get("image_paths", "").split("|")
        
        # Build interleaved image-text prompt
        messages = []
        for i, path in enumerate(image_paths):
            if path and Path(path).exists():
                messages.append({"type": "image", "value": path})
        
        messages.append({"type": "text", "value": line["question"]})
        
        return messages


class MOCHI_Grid(MOCHIBaseDataset):
    """
    MOCHI dataset with single grid image approach.
    
    Multiple images are merged into a 2x2 grid with labels A, B, C, D.
    This tests the model's ability to understand spatial layouts.
    """
    
    MODALITY = "image"
    
    def __init__(self, grid_size: int = 512):
        super().__init__("MOCHI_Grid")
        self.grid_size = grid_size
    
    def build_tsv(self, output_path: str, image_dir: Optional[str] = None):
        """Build TSV file with grid images."""
        hf_data = self.load_hf_dataset()
        
        if image_dir is None:
            image_dir = tempfile.mkdtemp(prefix="mochi_grid_")
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for idx, trial in enumerate(hf_data):
            images = trial["images"]
            n_objects = trial["n_objects"]
            oddity_index = trial["oddity_index"]
            answer = INDEX_TO_LETTER[oddity_index]
            
            # Create grid
            grid = create_labeled_grid(images, cell_size=self.grid_size)
            
            # Save grid image
            grid_path = image_dir / f"{trial['trial']}_grid.png"
            grid.save(grid_path)
            
            # Encode as base64
            img_b64 = image_to_base64(grid)
            
            # Build prompt
            if n_objects == 3:
                prompt = (
                    "This image shows a 2x2 grid with 3 objects labeled A, B, C (D is empty). "
                    "Each shows an object from a different viewpoint. "
                    "Two show the same 3D object, one is different (the odd-one-out). "
                    "Which is the odd-one-out? Answer with just: A, B, or C."
                )
                options = {"A": "Object A", "B": "Object B", "C": "Object C", "D": ""}
            else:
                prompt = (
                    "This image shows a 2x2 grid with 4 objects labeled A, B, C, D. "
                    "Each shows an object from a different viewpoint. "
                    "Three show the same 3D object, one is different (the odd-one-out). "
                    "Which is the odd-one-out? Answer with just: A, B, C, or D."
                )
                options = {"A": "Object A", "B": "Object B", "C": "Object C", "D": "Object D"}
            
            rows.append({
                "index": idx,
                "image": img_b64,
                "question": prompt,
                "answer": answer,
                **options,
                "n_objects": n_objects,
                "category": trial["condition"],
                "dataset_source": trial["dataset"],
                "trial_id": trial["trial"],
                "human_accuracy": trial["human_avg"],
                "image_path": str(grid_path),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Built MOCHI_Grid TSV with {len(rows)} samples at {output_path}")
        return output_path
    
    def build_prompt(self, line) -> list:
        """Build prompt for VLMEvalKit."""
        if isinstance(line, int):
            line = self._data.iloc[line]
        
        image_path = line.get("image_path", "")
        
        return [
            {"type": "image", "value": image_path},
            {"type": "text", "value": line["question"]},
        ]


# Registration function for VLMEvalKit
def register_mochi_datasets():
    """
    Register MOCHI datasets with VLMEvalKit.
    
    Call this from your custom dataset registration code.
    """
    return {
        "MOCHI_Naive": MOCHI_Naive,
        "MOCHI_Grid": MOCHI_Grid,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "grid", "both"], default="both")
    parser.add_argument("--output-dir", default="./mochi_data")
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode in ["naive", "both"]:
        ds_naive = MOCHI_Naive()
        ds_naive.build_tsv(
            output_dir / "MOCHI_Naive.tsv",
            image_dir=output_dir / "images_naive",
        )
    
    if args.mode in ["grid", "both"]:
        ds_grid = MOCHI_Grid()
        ds_grid.build_tsv(
            output_dir / "MOCHI_Grid.tsv",
            image_dir=output_dir / "images_grid",
        )
    
    print(f"\nMOCHI datasets built in {output_dir}")
