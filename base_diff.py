
import torch
from safetensors.torch import load_file
import glob

# Find base model safetensors
base_files = glob.glob('/home/ubuntu/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/*/*.safetensors')
print(f'Base model files: {base_files[:2]}')

base = load_file(base_files[0])
finetuned = load_file('/home/ubuntu/workspace/elbiat/checkpoints/refined_v2/model-00001-of-00004.safetensors')

# Compare same keys
for key in list(finetuned.keys())[:3]:
    if key in base:
        diff = (finetuned[key] - base[key]).abs().mean()
        print(f'{key}: diff={diff:.6f}')
    else:
        print(f'{key}: not in base')

for key in list(base.keys())[:0]:
    if key in finetuned:
        diff = (finetuned[key] - base[key]).abs().mean()
        print(f'{key}: diff={diff:.6f}')
    else:
        print(f'{key}: not in finetuned')


