"""
Multi-model VLM service supporting base, LoRA, VPT, and full fine-tuned models.

Config via MODEL_CONFIG env var pointing to a YAML file, or individual env vars.
"""

import os
import io
import yaml
import requests
from typing import Any, Optional, Dict
from pathlib import Path
import math
import gc

import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

import sys
INTERNVL_CHAT_DIR = os.environ.get(
    "INTERNVL_CHAT_DIR",
    "/home/ubuntu/workspace/elbiat/external/InternVL/internvl_chat",
)
sys.path.insert(0, INTERNVL_CHAT_DIR)
from internvl.model.internvl_chat import InternVLChatModel

import threading

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_BASE = os.environ.get("DATA_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL_CONFIG = os.environ.get("MODEL_CONFIG", None)
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "base")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Multi-Model VLM Service")


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class ModelEntry:
    def __init__(self, name: str, model: Any, tokenizer: Any, model_type: str, base_model: str, checkpoint: str = None):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type  # "base", "lora", "vpt", "full"
        self.base_model = base_model
        self.checkpoint = checkpoint
        self.lock = threading.Lock()

    def to_dict(self):
        return {
            "name": self.name,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "checkpoint": self.checkpoint,
            "status": "ready",
        }


model_registry: Dict[str, ModelEntry] = {}



# ---- preprocessing (official-style) ----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed


def image_bytes_to_tensor(data: bytes, input_size=448, max_num=12) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values


def _print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def _clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()



def _split_model(model_name):
    model_name = model_name.split("/")[-1]
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    
    return device_map




# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(checkpoint_path: str) -> str:
    """Find best subfolder (best > final > latest epoch > root)."""
    p = Path(checkpoint_path)
    if not p.exists():
        return checkpoint_path
    if (p / "best").exists():
        return str(p / "best")
    if (p / "final").exists():
        return str(p / "final")
    epoch_folders = sorted(
        [f for f in p.iterdir() if f.is_dir() and f.name.startswith("epoch_")],
        key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
        reverse=True
    )
    if epoch_folders:
        return str(epoch_folders[0])
    return checkpoint_path






def is_lora_checkpoint(path: str) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def load_base_model(base_model_name: str):
    """Load a base HuggingFace model."""
    device_map = _split_model(base_model_name)
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    if DEVICE == "cuda" and device_map == "auto":
        model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_lora_model(base_model_name: str, lora_path: str):
    """Load base model with LoRA adapter merged."""
    from peft import PeftModel
    
    checkpoint = find_best_checkpoint(lora_path)
    print(f"  Loading LoRA from: {checkpoint}")
    
    device_map = _split_model(base_model_name)
    base_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    )
    
    # Adapter was trained on the LLM alone (keys: base_model.model.model.layers...),
    # so attach it to .language_model, not the whole VLM, just like the eval code
    base_model.language_model = PeftModel.from_pretrained(base_model.language_model, checkpoint)
    base_model.language_model = base_model.language_model.merge_and_unload()
    model = base_model.eval()
    
    if DEVICE == "cuda" and device_map == "auto":
        model = model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer, checkpoint


def load_full_model(model_path: str, base_model_name: str):
    """Load a fully fine-tuned model."""
    checkpoint = find_best_checkpoint(model_path)
    print(f"  Loading full model from: {checkpoint}")
    
    device_map = _split_model(base_model_name)

    model = InternVLChatModel.from_pretrained(
        checkpoint,
        torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    
    if DEVICE == "cuda":
        model = model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    return model, tokenizer, checkpoint


def load_model_config() -> dict:
    """Load model configuration from YAML or env vars."""
    if MODEL_CONFIG and Path(MODEL_CONFIG).exists():
        with open(MODEL_CONFIG) as f:
            return yaml.safe_load(f)
    
    # Default: just load base model from env
    base = os.environ.get("INTERNVL_MODEL", "OpenGVLab/InternVL2_5-2B")
    config = {"models": [{"name": "base", "type": "base", "base_model": base}]}
    
    # Check for additional models via env vars
    # LORA_MODELS="name1:path1,name2:path2"
    lora_models = os.environ.get("LORA_MODELS", "")
    if lora_models:
        for entry in lora_models.split(","):
            if ":" in entry:
                name, path = entry.split(":", 1)
                config["models"].append({
                    "name": name.strip(),
                    "type": "lora",
                    "base_model": base,
                    "checkpoint": path.strip(),
                })
    
    return config


@app.on_event("startup")
def load_models():
    """Load all configured models at startup."""
    global model_registry
    
    config = load_model_config()
    print(f"Loading {len(config['models'])} model(s)...")
    
    for model_cfg in config["models"]:
        name = model_cfg["name"]
        model_type = model_cfg.get("type", "base")
        base_model = model_cfg.get("base_model", "OpenGVLab/InternVL2_5-2B")
        checkpoint = model_cfg.get("checkpoint")
        
        print(f"\nLoading '{name}' ({model_type})...")
        
        try:
            if model_type == "base":
                model, tokenizer = load_base_model(base_model)
                entry = ModelEntry(name, model, tokenizer, model_type, base_model)
            
            elif model_type == "lora":
                model, tokenizer, ckpt = load_lora_model(base_model, checkpoint)
                entry = ModelEntry(name, model, tokenizer, model_type, base_model, ckpt)
            
            elif model_type == "full":
                model, tokenizer, ckpt = load_full_model(checkpoint, base_model)
                entry = ModelEntry(name, model, tokenizer, model_type, base_model, ckpt)
            
            else:
                print(f"  Unknown model type: {model_type}, skipping")
                continue
            
            model_registry[name] = entry
            print(f"  ✓ Loaded '{name}'")
            
        except Exception as e:
            print(f"  ✗ Failed to load '{name}': {e}")

    if "base" in model_registry and "dpo-v1" in model_registry:
        wb = model_registry["base"].model.language_model.model.layers[0].attention.wo.weight
        wd = model_registry["dpo-v1"].model.language_model.model.layers[0].attention.wo.weight
        print("[check] base vs dpo-v1 layer0 wo max diff:", (wb.float() - wd.float()).abs().max().item())
     
    print(f"\n{len(model_registry)} model(s) ready")


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models():
    """List all available models."""
    return {
        "default": DEFAULT_MODEL,
        "models": [entry.to_dict() for entry in model_registry.values()]
    }


class ChatReq(BaseModel):
    prompt: str
    image_id: Optional[int] = None
    history: Optional[Any] = None
    max_new_tokens: int = 256
    do_sample: bool = False
    max_num_tiles: int = 12
    return_history: bool = True


@app.post("/chat")
def chat(
    req: ChatReq,
    model: str = Query(default=None, description="Model name to use (default: base)")
):
    """Run chat inference with specified model."""
    model_name = model or DEFAULT_MODEL
    
    if model_name not in model_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(model_registry.keys())}"
        )
    
    entry = model_registry[model_name]
    
    pixel_values = None
    if req.image_id is not None:
        url = f"{DATA_BASE}/images/{req.image_id}/file"
        r = requests.get(url, timeout=30)
        if not r.ok:
            raise HTTPException(status_code=404, detail=f"image_id {req.image_id} not found")
        pixel_values = image_bytes_to_tensor(r.content, max_num=req.max_num_tiles)
        if DEVICE == "cuda":
            pixel_values = pixel_values.to(DTYPE).cuda()

    gen_cfg = {"max_new_tokens": req.max_new_tokens, "do_sample": req.do_sample}

    with torch.no_grad():
        with entry.lock:
            out = entry.model.chat(
                entry.tokenizer, pixel_values, req.prompt, gen_cfg,
                history=req.history, return_history=req.return_history
            )

    if req.return_history:
        response, history = out
        return {"model": model_name, "response": response, "history": history}
    return {"model": model_name, "response": out}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(model_registry)}
