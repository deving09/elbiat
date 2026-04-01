# ~/workspace/elbiat/scripts/train_vpt.py
"""
Train Visual Prompt Tuning for InternVL.

Usage:
    python scripts/train_vpt.py \
        --train-data feedback_data/refined_v1/train.jsonl \
        --output-dir checkpoints/vpt_v1 \
        --num-tokens 10 \
        --mode deep \
        --epochs 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'external' / 'InternVL' / 'internvl_chat'))

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from tqdm import tqdm
from PIL import Image

from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from vpt.vpt_internvl import VPTInternVL


class VPTDataset(Dataset):
    """Dataset for VPT training."""
    
    def __init__(self, jsonl_path: str, root_dir: str, image_size: int = 448, max_num: int = 6):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.max_num = max_num
        self.transform = build_transform(is_train=True, input_size=image_size)
        
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.root_dir / sample['image']
        image = Image.open(image_path).convert('RGB')
        
        # Dynamic preprocessing
        images = dynamic_preprocess(image, image_size=self.image_size, max_num=self.max_num)
        pixel_values = torch.stack([self.transform(img) for img in images])
        
        # Extract text from conversations
        prompt = ""
        response = ""
        for turn in sample.get('conversations', []):
            if turn['from'] == 'human':
                prompt = turn['value'].replace('<image>\n', '').strip()
            elif turn['from'] == 'gpt':
                response = turn['value'].strip()
        
        return {
            'pixel_values': pixel_values,
            'prompt': prompt,
            'response': response,
        }


def collate_fn(batch):
    """Collate function handling variable image counts."""
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    
    # Track which images belong to which sample
    image_counts = [item['pixel_values'].shape[0] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'prompts': prompts,
        'responses': responses,
        'image_counts': image_counts,
    }


def train_vpt(args):
    print("=" * 60)
    print("Visual Prompt Tuning for InternVL")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    base_model = InternVLChatModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Wrap with VPT
    print(f"\nInitializing VPT with {args.num_tokens} tokens, mode={args.mode}")
    model = VPTInternVL(
        base_model=base_model,
        num_tokens=args.num_tokens,
        mode=args.mode,
        dropout=args.dropout,
    )
    model.base_model.vision_model.encoder.gradient_checkpointing = False
    model = model.to(device)
    
    trainable_params = model.get_num_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    # Dataset
    print(f"\nLoading training data: {args.train_data}")
    dataset = VPTDataset(
        jsonl_path=args.train_data,
        root_dir=args.root_dir,
        image_size=448,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Training samples: {len(dataset)}")
    
    # Optimizer
    optimizer = AdamW(
        model.vpt_vision.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    num_training_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    # Loss function - we'll use a simple contrastive/matching loss
    # For full training, you'd want to compute LM loss through the full model
    criterion = nn.CosineEmbeddingLoss()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            
            # Forward through VPT vision encoder
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Get projected vision features
                vision_features = model.forward_vision(pixel_values)
                
                # Simple contrastive loss between samples in batch
                # Normalize features
                feat_norm = torch.nn.functional.normalize(vision_features.mean(dim=1), dim=-1)
                
                # Cosine similarity matrix
                sim_matrix = torch.matmul(feat_norm, feat_norm.T)
                
                # Contrastive loss - encourage diversity
                batch_size = feat_norm.shape[0]
                labels = torch.arange(batch_size, device=device)
                loss = torch.nn.functional.cross_entropy(sim_matrix / 0.07, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vpt_vision.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_vpt(output_dir / 'best')
        
        model.save_vpt(output_dir / f'epoch_{epoch+1}')
    
    # Save final
    model.save_vpt(output_dir / 'final')
    print(f"\nTraining complete. Models saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='OpenGVLab/InternVL2_5-2B')
    parser.add_argument('--train-data', required=True, help='Path to training JSONL')
    parser.add_argument('--root-dir', default='/home/ubuntu/workspace/elbiat', help='Root dir for image paths')
    parser.add_argument('--output-dir', default='checkpoints/vpt_v1')
    parser.add_argument('--num-tokens', type=int, default=10, help='Number of prompt tokens')
    parser.add_argument('--mode', choices=['shallow', 'deep'], default='deep')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    
    args = parser.parse_args()
    train_vpt(args)


if __name__ == '__main__':
    main()