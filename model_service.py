import os
import io
import requests
from typing import Any, Optional
import math

import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

import threading


DATA_BASE = os.environ.get("DATA_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL_NAME = os.environ.get("INTERNVL_MODEL", "OpenGVLab/InternVL2_5-2B")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Model Service (InternVL2.5-2B)")


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







# ---- load model once ----
model = None
tokenizer = None
gpu_lock = threading.Lock()



@app.on_event("startup")
def load_model():
    global model, tokenizer, gpu_lock
    world_size = torch.cuda.device_count()
    device_map = _split_model(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    if DEVICE == "cuda":
        model = model.cuda()
        #gpu_lock = torch.cuda.Lock()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)



class ChatReq(BaseModel):
    prompt: str
    image_id: Optional[int] = None
    history: Optional[Any] = None
    max_new_tokens: int = 256
    do_sample: bool = False
    max_num_tiles: int = 12
    return_history: bool = True

@app.post("/chat/internvl2_5_2b")
def chat(req: ChatReq):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    pixel_values = None
    if req.image_id is not None:
        # fetch image bytes from data service
        url = f"{DATA_BASE}/images/{req.image_id}/file"
        r = requests.get(url, timeout=30)
        if not r.ok:
            raise HTTPException(status_code=404, detail=f"image_id {req.image_id} not found in data service")
        pixel_values = image_bytes_to_tensor(r.content, max_num=req.max_num_tiles)
        if DEVICE == "cuda":
            pixel_values = pixel_values.to(DTYPE).cuda()

    gen_cfg = {"max_new_tokens": req.max_new_tokens, "do_sample": req.do_sample}

    with torch.no_grad():
        if DEVICE == "cuda":
            with gpu_lock:
                out = model.chat(tokenizer, pixel_values, req.prompt, gen_cfg,
                                 history=req.history, return_history=req.return_history)
        else:
            out = model.chat(tokenizer, pixel_values, req.prompt, gen_cfg,
                             history=req.history, return_history=req.return_history)

    if req.return_history:
        response, history = out
        return {"model": MODEL_NAME, "response": response, "history": history}
    return {"model": MODEL_NAME, "response": out}
