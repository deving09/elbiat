import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

model_path = '/home/ubuntu/workspace/elbiat/checkpoints/refined_v2'

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='auto'
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

image = Image.open('/home/ubuntu/workspace/elbiat/examples/hidden_frog.jpg').convert('RGB')
pixel_values = model.vis_processor(image).unsqueeze(0).to(model.device)

question = 'Describe this image.'
response = model.chat(tokenizer, pixel_values, question, generation_config=dict(max_new_tokens=512, do_sample=False))

print('Response:', response)

