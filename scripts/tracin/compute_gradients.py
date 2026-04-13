"""
Compute and cache gradients for TracIn attribution.

For each example (train or test), we compute the gradient of the loss
with respect to model parameters and store it as a flat vector.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Literal
from dataclasses import dataclass

# Add paths
sys.path.insert(0, '/home/ubuntu/workspace/elbiat/external/InternVL/internvl_chat')

from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess

from transformers import AutoTokenizer
from PIL import Image

from internvl.conversation import get_conv_template

import torch.nn.functional as F


@dataclass
class GradientConfig:
    model_path: str
    output_dir: str
    batch_size: int = 1  # Gradient computation is memory intensive
    max_length: int = 512
    image_size: int = 448
    gradient_layers: str = "lora"  # "lora", "llm", "all"


def get_parameter_subset(model, layer_type: str = "lora"):
    """Get subset of parameters to compute gradients for."""
    params = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if layer_type == "lora":
            if "lora_" in name:
                params[name] = param
        elif layer_type == "llm":
            if "language_model" in name:
                params[name] = param
        elif layer_type == "all":
            params[name] = param
    
    return params




def compute_example_gradient(
    model,
    tokenizer,
    transform,
    dynamic_preprocess_fn,
    image_path: str,
    question: str,
    answer: str,
    target_params: dict,
    device: str = "cuda",
) -> np.ndarray:



    model.zero_grad()

    image = Image.open(image_path).convert("RGB")
    patches = dynamic_preprocess_fn(image, image_size=448, max_num=6)
    pixel_values = torch.stack([transform(p) for p in patches]).to(device, dtype=torch.bfloat16)

    vit_embeds = model.extract_feature(pixel_values)   # [num_img_tokens, hidden] or similar

    # Flatten to [1, num_patches * 256, hidden]
    vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
    #if vit_embeds.dim() == 2:
    #    vit_embeds = vit_embeds.unsqueeze(0)           # [1, N, C]

    # build text the same way every time
    full_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    full_tokens = tokenizer(full_text, return_tensors="pt")
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")

    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)

    labels = input_ids.clone()
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    input_embeds = model.language_model.get_input_embeddings()(input_ids)

    B, N, C = vit_embeds.shape
    input_embeds = torch.cat([vit_embeds, input_embeds], dim=1)

    vision_mask = torch.ones((1, N), device=device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

    outputs = model.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )

    logits = outputs.logits[:, N:, :]   # strip vision positions

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    """
    print("loss.requires_grad =", loss.requires_grad)
    print(f"loss: {loss}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"labels non-masked count: {(labels != -100).sum()}")
    print(f"shift labels len: {shift_labels.shape[-1]}")
    """
    loss.backward()

    grads = []
    for name, param in target_params.items():
        grads.append(
            param.grad.detach().cpu().float().flatten()
            if param.grad is not None else
            torch.zeros(param.numel())
        )

    return torch.cat(grads).numpy()




def compute_gradients_for_dataset(
    config: GradientConfig,
    examples: list[dict],  # [{"id": ..., "image": ..., "question": ..., "answer": ...}, ...]
    split: Literal["train", "test"],
    benchmark: Optional[str] = None,
):
    """
    Compute gradients for all examples in a dataset.
    
    Saves gradients to: {output_dir}/{split}_{benchmark}_gradients.npz
    """
    print(f"Loading model: {config.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=False
    )
    

    #device = "cuda"

    model = InternVLChatModel.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()

    if hasattr(model, "vision_model") and hasattr(model, "vision_model") and hasattr(model.vision_model, "gradient_checkpointing_disable"):
        model.vision_model.gradient_checkpointing_disable()

    # Set the img_context_token_id
    img_context_token = '<IMG_CONTEXT>'
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)


    model.train()  # Use train mode for gradient computation

    # Ensure all target parameters require grad
    for name, param in model.named_parameters():
        if "lora_" in name or config.gradient_layers == "all":
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    # Check what's trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params: {len(trainable)}")
    print(f"First few: {trainable[:5]}")
    
    transform = build_transform(is_train=True, input_size=config.image_size)
    
    target_params = get_parameter_subset(model, config.gradient_layers)
    print(f"Computing gradients for {len(target_params)} parameter groups")


    target_params = get_parameter_subset(model, config.gradient_layers)
    print(f"Target params: {len(target_params)}")
    print(f"Param names: {list(target_params.keys())[:5]}")  # First 5
    
    # Compute total gradient dimension
    total_dim = sum(p.numel() for p in target_params.values())
    print(f"Gradient dimension: {total_dim:,}")
    
    # Storage
    gradients = []
    example_ids = []
    
    for example in tqdm(examples, desc=f"Computing {split} gradients"):
        try:
            grad = compute_example_gradient(
                model=model,
                tokenizer=tokenizer,
                transform=transform,
                dynamic_preprocess_fn=dynamic_preprocess,
                image_path=example["image"],
                question=example["question"],
                answer=example["answer"],
                target_params=target_params,
            )
            gradients.append(grad)
            example_ids.append(example["id"])
        except Exception as e:
            import traceback
            print(f"Error on example {example['id']}: {e}")
            traceback.print_exc()
            continue
    
    # Save
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{benchmark}" if benchmark else ""
    output_file = output_dir / f"{split}{suffix}_gradients.npz"
    
    np.savez_compressed(
        output_file,
        gradients=np.stack(gradients),
        example_ids=np.array(example_ids),
    )


    # At the end of compute_gradients_for_dataset, before np.savez:
    if len(gradients) == 0:
        raise ValueError(f"All {len(examples)} examples failed. Check error messages above.") 
    
    print(f"Saved {len(gradients)} gradients to {output_file}")
    return output_file
