"""
MOCHI Dataset wrapper for VLM evaluation.

MOCHI (Multiview Object Consistency in Humans and Image models) is a benchmark
for evaluating 3D shape understanding. Each trial shows 3-4 views of objects
and the task is to identify the "odd-one-out" - the object that doesn't match.

This module provides two evaluation approaches:
1. Naive: Pass multiple images separately to the VLM
2. Single Image: Merge images into a labeled grid (A, B, C, D)

Paper: https://arxiv.org/abs/2409.05862
Dataset: https://huggingface.co/datasets/tzler/MOCHI
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class MOCHIConfig:
    """Configuration for MOCHI dataset."""
    cache_dir: str = "~/.cache/mochi"
    image_size: int = 448  # Resize images to this size
    grid_size: int = 512   # Size of each cell in grid mode
    font_size: int = 48    # Font size for labels in grid mode


# Index to letter mapping
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def download_mochi(cache_dir: str = "~/.cache/mochi") -> pd.DataFrame:
    """
    Download MOCHI dataset from HuggingFace.
    
    Returns:
        DataFrame with all trial data
    """
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading MOCHI dataset from HuggingFace...")
    dataset = load_dataset("tzler/MOCHI", split="train")
    
    return dataset


def create_labeled_grid(
    images: list[Image.Image],
    cell_size: int = 512,
    font_size: int = 48,
    background_color: tuple = (255, 255, 255),
    label_color: tuple = (0, 0, 0),
) -> Image.Image:
    """
    Create a 2x2 grid of images with labels A, B, C, D.
    
    Args:
        images: List of 3 or 4 PIL images
        cell_size: Size of each cell in pixels
        font_size: Font size for labels
        background_color: Background color (RGB)
        label_color: Label text color (RGB)
    
    Returns:
        Combined grid image
    """
    n_images = len(images)
    assert n_images in [3, 4], f"Expected 3 or 4 images, got {n_images}"
    
    # Create 2x2 grid
    grid_width = cell_size * 2
    grid_height = cell_size * 2
    grid = Image.new("RGB", (grid_width, grid_height), background_color)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    # Positions for 2x2 grid: A(top-left), B(top-right), C(bottom-left), D(bottom-right)
    positions = [
        (0, 0),           # A
        (cell_size, 0),   # B
        (0, cell_size),   # C
        (cell_size, cell_size),  # D
    ]
    
    draw = ImageDraw.Draw(grid)
    
    for i, (img, (x, y)) in enumerate(zip(images, positions)):
        # Resize image to fit cell (with some padding for label)
        img_size = cell_size - font_size - 20  # Leave space for label
        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        
        # Paste image (centered horizontally, below label)
        paste_x = x + (cell_size - img_size) // 2
        paste_y = y + font_size + 10
        grid.paste(img_resized, (paste_x, paste_y))
        
        # Draw label
        label = INDEX_TO_LETTER[i]
        label_x = x + cell_size // 2
        label_y = y + 5
        draw.text((label_x, label_y), label, fill=label_color, font=font, anchor="mt")
    
    # If only 3 images, mark D cell as empty
    if n_images == 3:
        x, y = positions[3]
        center_x = x + cell_size // 2
        center_y = y + cell_size // 2
        draw.text((center_x, center_y), "(empty)", fill=(128, 128, 128), font=font, anchor="mm")
    
    return grid


class MOCHIDataset:
    """
    MOCHI dataset wrapper for VLM evaluation.
    
    Supports two modes:
    - naive: Returns multiple separate images
    - single_image: Returns a merged grid with labels
    """
    
    def __init__(
        self,
        mode: Literal["naive", "single_image"] = "naive",
        cache_dir: str = "~/.cache/mochi",
        subset: Optional[str] = None,  # Filter by dataset (e.g., "barense", "shapegen")
        condition: Optional[str] = None,  # Filter by condition
        max_samples: Optional[int] = None,
    ):
        """
        Initialize MOCHI dataset.
        
        Args:
            mode: "naive" for multi-image, "single_image" for grid
            cache_dir: Directory to cache dataset
            subset: Filter by dataset name
            condition: Filter by condition
            max_samples: Limit number of samples
        """
        self.mode = mode
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = load_dataset("tzler/MOCHI", split="train")
        
        # Apply filters
        if subset:
            self.dataset = self.dataset.filter(lambda x: x["dataset"] == subset)
        if condition:
            self.dataset = self.dataset.filter(lambda x: x["condition"] == condition)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} MOCHI trials (mode={mode})")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a trial.
        
        Returns:
            dict with:
                - id: trial identifier
                - images: list of PIL images (naive) or single grid image (single_image)
                - n_objects: number of objects (3 or 4)
                - answer: correct answer letter (A, B, C, or D)
                - answer_index: correct answer index (0-3)
                - prompt: question prompt
                - metadata: additional trial info
        """
        trial = self.dataset[idx]
        
        images = trial["images"]
        n_objects = trial["n_objects"]
        oddity_index = trial["oddity_index"]
        answer_letter = INDEX_TO_LETTER[oddity_index]
        
        if self.mode == "naive":
            # Return separate images
            prompt = self._build_naive_prompt(n_objects)
            return_images = images
        else:
            # Create grid
            grid = create_labeled_grid(images)
            prompt = self._build_grid_prompt(n_objects)
            return_images = grid
        
        return {
            "id": trial["trial"],
            "images": return_images,
            "n_objects": n_objects,
            "answer": answer_letter,
            "answer_index": oddity_index,
            "prompt": prompt,
            "metadata": {
                "dataset": trial["dataset"],
                "condition": trial["condition"],
                "human_avg": trial["human_avg"],
                "human_std": trial["human_std"],
            },
        }
    
    def _build_naive_prompt(self, n_objects: int) -> str:
        """Build prompt for naive multi-image mode."""
        if n_objects == 3:
            return (
                "You are shown 3 images labeled Image 1, Image 2, and Image 3. "
                "Each image shows an object from a different viewpoint. "
                "Two of these images show the same 3D object from different views, "
                "while one image shows a DIFFERENT object (the odd-one-out). "
                "Which image shows the odd-one-out? "
                "Answer with just the letter: A (Image 1), B (Image 2), or C (Image 3)."
            )
        else:
            return (
                "You are shown 4 images labeled Image 1, Image 2, Image 3, and Image 4. "
                "Each image shows an object from a different viewpoint. "
                "Three of these images show the same 3D object from different views, "
                "while one image shows a DIFFERENT object (the odd-one-out). "
                "Which image shows the odd-one-out? "
                "Answer with just the letter: A (Image 1), B (Image 2), C (Image 3), or D (Image 4)."
            )
    
    def _build_grid_prompt(self, n_objects: int) -> str:
        """Build prompt for single image grid mode."""
        if n_objects == 3:
            return (
                "This image shows a 2x2 grid with 3 objects labeled A, B, and C "
                "(cell D is empty). Each shows an object from a different viewpoint. "
                "Two of these show the same 3D object from different views, "
                "while one shows a DIFFERENT object (the odd-one-out). "
                "Which one is the odd-one-out? Answer with just the letter: A, B, or C."
            )
        else:
            return (
                "This image shows a 2x2 grid with 4 objects labeled A, B, C, and D. "
                "Each shows an object from a different viewpoint. "
                "Three of these show the same 3D object from different views, "
                "while one shows a DIFFERENT object (the odd-one-out). "
                "Which one is the odd-one-out? Answer with just the letter: A, B, C, or D."
            )
    
    def to_vlmeval_tsv(self, output_path: str, image_dir: str) -> str:
        """
        Export dataset to VLMEvalKit TSV format.
        
        Args:
            output_path: Path to save TSV file
            image_dir: Directory to save images
        
        Returns:
            Path to TSV file
        """
        import base64
        from io import BytesIO
        
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for idx in tqdm(range(len(self)), desc="Exporting to TSV"):
            item = self[idx]
            
            if self.mode == "naive":
                # For naive mode, we need to handle multiple images
                # VLMEvalKit doesn't natively support this, so we'll save paths
                image_paths = []
                for i, img in enumerate(item["images"]):
                    img_path = image_dir / f"{item['id']}_{i}.png"
                    img.save(img_path)
                    image_paths.append(str(img_path))
                
                # Encode first image for TSV (others referenced in question)
                buffered = BytesIO()
                item["images"][0].save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Store additional image paths in question
                extra_images = "|".join(image_paths[1:])
            else:
                # Single image mode
                img_path = image_dir / f"{item['id']}_grid.png"
                item["images"].save(img_path)
                
                buffered = BytesIO()
                item["images"].save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                extra_images = ""
            
            # Build options
            if item["n_objects"] == 3:
                options = ["A", "B", "C"]
            else:
                options = ["A", "B", "C", "D"]
            
            row = {
                "index": idx,
                "image": img_b64,
                "question": item["prompt"],
                "answer": item["answer"],
                "A": "Image A is the odd-one-out",
                "B": "Image B is the odd-one-out",
                "C": "Image C is the odd-one-out",
                "D": "Image D is the odd-one-out" if item["n_objects"] == 4 else "",
                "category": item["metadata"]["condition"],
                "extra_images": extra_images,  # For naive mode
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep="\t", index=False)
        
        print(f"Exported {len(rows)} samples to {output_path}")
        return output_path


def parse_answer(response: str, n_objects: int = 4) -> Optional[str]:
    """
    Parse model response to extract answer letter.
    
    Args:
        response: Model's response text
        n_objects: Number of objects (3 or 4)
    
    Returns:
        Answer letter (A, B, C, D) or None if parsing failed
    """
    response = response.strip().upper()
    
    valid_answers = ["A", "B", "C"] if n_objects == 3 else ["A", "B", "C", "D"]
    
    # Check for direct single letter answer
    if response in valid_answers:
        return response
    
    # Check for "Image X" or "Option X" pattern
    for letter in valid_answers:
        patterns = [
            f"IMAGE {letter}",
            f"OPTION {letter}",
            f"ANSWER: {letter}",
            f"ANSWER IS {letter}",
            f"({letter})",
            f"[{letter}]",
            f"{letter}.",
            f"{letter})",
            f"THE ODD-ONE-OUT IS {letter}",
            f"THE ODD ONE OUT IS {letter}",
        ]
        for pattern in patterns:
            if pattern in response:
                return letter
    
    # Check if any single letter appears
    for letter in valid_answers:
        if f" {letter} " in f" {response} " or response.startswith(letter) or response.endswith(letter):
            return letter
    
    return None


if __name__ == "__main__":
    # Test the dataset
    print("Testing MOCHI Dataset...")
    
    # Test naive mode
    print("\n=== Naive Mode ===")
    ds_naive = MOCHIDataset(mode="naive", max_samples=5)
    sample = ds_naive[0]
    print(f"Trial: {sample['id']}")
    print(f"N objects: {sample['n_objects']}")
    print(f"Answer: {sample['answer']} (index {sample['answer_index']})")
    print(f"N images: {len(sample['images'])}")
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Human accuracy: {sample['metadata']['human_avg']:.2%}")
    
    # Test single image mode
    print("\n=== Single Image Mode ===")
    ds_grid = MOCHIDataset(mode="single_image", max_samples=5)
    sample = ds_grid[0]
    print(f"Trial: {sample['id']}")
    print(f"Grid image size: {sample['images'].size}")
    print(f"Prompt: {sample['prompt'][:100]}...")
    
    # Save sample grid
    sample["images"].save("/tmp/mochi_grid_sample.png")
    print(f"Saved sample grid to /tmp/mochi_grid_sample.png")
