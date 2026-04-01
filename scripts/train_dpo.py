"""
DPO training for InternVL.

Usage:
    python scripts/train_dpo.py \
        --train-data feedback_data/dpo_v1/train.jsonl \
        --output-dir checkpoints/dpo_v1 \
        --epochs 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "InternVL" / "internvl_chat"))

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess


class DPODataset(Dataset):
    """Dataset for DPO training."""
    
    def __init__(
        self,
        jsonl_path: str,
        root_dir: str,
        tokenizer,
        image_size: int = 448,
        max_num: int = 6,
        max_length: int = 2048,
    ):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_num = max_num
        self.max_length = max_length
        self.transform = build_transform(is_train=False, input_size=image_size)
        
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} DPO samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _tokenize(self, prompt: str, response: str):
        """Tokenize prompt + response."""
        full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        full_tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])
        
        # Create labels: -100 for prompt, actual ids for response
        labels = full_tokens["input_ids"].clone().squeeze(0)
        labels[:prompt_len] = -100
        
        return {
            "input_ids": full_tokens["input_ids"].squeeze(0),
            "attention_mask": full_tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.root_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        
        images = dynamic_preprocess(image, image_size=self.image_size, max_num=self.max_num)
        pixel_values = torch.stack([self.transform(img) for img in images])
        
        prompt = sample["prompt"]
        chosen_tokens = self._tokenize(prompt, sample["chosen"])
        rejected_tokens = self._tokenize(prompt, sample["rejected"])
        
        return {
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "chosen_labels": chosen_tokens["labels"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
            "rejected_labels": rejected_tokens["labels"],
        }


def dpo_collate_fn(batch):
    """Collate function for DPO batches."""
    # For simplicity, we process one sample at a time due to variable image counts
    # Stack everything
    return {
        "pixel_values": [item["pixel_values"] for item in batch],
        "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
        "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
        "chosen_labels": torch.stack([item["chosen_labels"] for item in batch]),
        "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
        "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch]),
        "rejected_labels": torch.stack([item["rejected_labels"] for item in batch]),
    }


def compute_logps(model, input_ids, attention_mask, labels, pixel_values, device):
    """Compute log probabilities for sequences."""
    # Concatenate pixel values for batch
    all_pixel_values = torch.cat(pixel_values, dim=0).to(device, dtype=torch.bfloat16)
    
    # Get image embeds
    vit_embeds = model.extract_feature(all_pixel_values)
    
    # Split back to per-sample
    image_counts = [pv.shape[0] for pv in pixel_values]
    vit_embeds_list = torch.split(vit_embeds, image_counts, dim=0)
    
    batch_logps = []
    
    for i in range(len(pixel_values)):
        sample_input_ids = input_ids[i:i+1].to(device)
        sample_attention_mask = attention_mask[i:i+1].to(device)
        sample_labels = labels[i:i+1].to(device)
        #sample_vit_embeds = vit_embeds_list[i].unsqueeze(0)

        
        sample_vit_embeds = vit_embeds_list[i]

        # Ensure 3D shape [1, num_tokens, hidden_dim]
        if sample_vit_embeds.dim() == 2:
            sample_vit_embeds = sample_vit_embeds.unsqueeze(0)
        elif sample_vit_embeds.dim() == 3 and sample_vit_embeds.shape[0] != 1:
            sample_vit_embeds = sample_vit_embeds.view(1, -1, sample_vit_embeds.shape[-1]) 
        
        # Get input embeds
        input_embeds = model.language_model.get_input_embeddings()(sample_input_ids)
        
        # Find image token position and insert vision embeds
        # For simplicity, prepend vision embeds
        B, N, C = sample_vit_embeds.shape

        input_embeds = torch.cat([
            sample_vit_embeds.view(1, -1, C),
            input_embeds
        ], dim=1)
        
        # Extend attention mask
        vision_mask = torch.ones(1, N, device=device, dtype=sample_attention_mask.dtype)
        extended_attention_mask = torch.cat([vision_mask, sample_attention_mask], dim=1)
        
        # Forward
        outputs = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )
        
        # Get logits for text part only
        logits = outputs.logits[:, N:, :]  # Remove vision token positions
        
        # Compute log probs
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sample_labels[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)
        
        # Mask out ignored tokens (-100)
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask
        
        # Average log prob
        seq_logp = token_log_probs.sum() / mask.sum().clamp(min=1)
        batch_logps.append(seq_logp)
    
    return torch.stack(batch_logps)


def dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss."""
    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
    
    losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
    
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    margin = (chosen_rewards - rejected_rewards).mean()
    
    return losses.mean(), accuracy.item(), margin.item()


def train_dpo(args):
    print("=" * 60)
    print("DPO Training for InternVL")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy model
    print(f"Loading policy model: {args.base_model}")
    policy_model = InternVLChatModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Add LoRA
    print(f"Adding LoRA (rank={args.lora_rank})")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["wqkv", "wo", "w1", "w2", "w3"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model.language_model = get_peft_model(policy_model.language_model, lora_config)
    policy_model.language_model.print_trainable_parameters()
    
    # Freeze vision
    for param in policy_model.vision_model.parameters():
        param.requires_grad = False
    for param in policy_model.mlp1.parameters():
        param.requires_grad = False
    
    policy_model = policy_model.to(device)
    
    # Reference model
    print("Creating reference model...")
    ref_model = InternVLChatModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Dataset
    print(f"\nLoading training data: {args.train_data}")
    train_dataset = DPODataset(
        jsonl_path=args.train_data,
        root_dir=args.root_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=dpo_collate_fn,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    num_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    
    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Beta: {args.beta}, LR: {args.lr}")
    
    best_accuracy = 0
    
    for epoch in range(args.epochs):
        policy_model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Policy log probs
                policy_chosen_logps = compute_logps(
                    policy_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                    batch["pixel_values"],
                    device
                )
                policy_rejected_logps = compute_logps(
                    policy_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                    batch["pixel_values"],
                    device
                )
                
                # Reference log probs
                with torch.no_grad():
                    ref_chosen_logps = compute_logps(
                        ref_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                        batch["pixel_values"],
                        device
                    )
                    ref_rejected_logps = compute_logps(
                        ref_model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                        batch["pixel_values"],
                        device
                    )
                
                loss, acc, margin = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=args.beta
                )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2%}"})
        
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={avg_acc:.2%}")
        
        # Save
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            save_path = output_dir / "best"
            save_path.mkdir(exist_ok=True)
            policy_model.language_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved best model (acc={avg_acc:.2%})")
    
    # Final save
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    policy_model.language_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\nDone! Best accuracy: {best_accuracy:.2%}")
    print(f"Models saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="OpenGVLab/InternVL2_5-2B")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--root-dir", default="/home/ubuntu/workspace/elbiat")
    parser.add_argument("--output-dir", default="checkpoints/dpo_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    
    args = parser.parse_args()
    train_dpo(args)


if __name__ == "__main__":
    main()
