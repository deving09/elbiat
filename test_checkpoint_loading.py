import sys
sys.path.insert(0, '/home/ubuntu/workspace/elbiat/external/InternVL/internvl_chat')

import math
import torch
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.internvl_chat import InternVLChatConfig
from PIL import Image
from internvl.train.dataset import build_transform, dynamic_preprocess

checkpoint = '/home/ubuntu/workspace/elbiat/checkpoints/refined_v1'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)

print('Loading model...')
model = InternVLChatModel.from_pretrained(
    checkpoint,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
).eval().cuda()

print('Loading image...')
image = Image.open('/home/ubuntu/workspace/elbiat/examples/hidden_frog.jpg').convert('RGB')
images = dynamic_preprocess(image, image_size=448, max_num=6)
transform = build_transform(is_train=False, input_size=448)
pixel_values = torch.stack([transform(img) for img in images]).to(torch.bfloat16).cuda()

print(f'Pixel values shape: {pixel_values.shape}')

print('Generating...')
generation_config = dict(max_new_tokens=512, do_sample=False)
response = model.chat(tokenizer, pixel_values, 'Describe this image.', generation_config)

print('='*50)
print('Response:', response)
print('='*50)