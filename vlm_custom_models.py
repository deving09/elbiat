# Custom model registration for finetuned models
import sys
import os
import warnings
sys.path.insert(0, '/home/ubuntu/workspace/elbiat/external/InternVL/internvl_chat')
sys.path.insert(0, '/home/ubuntu/workspace/elbiat/external/VLMEvalKit')


from functools import partial
#from vlmeval.vlm import InternVLChat
import torch
from vlmeval.vlm.internvl.internvl_chat import InternVLChat
from internvl.model.internvl_chat import InternVLChatModel

from transformers import AutoTokenizer, AutoModel

from peft import PeftModel

class DPOInternVLChat(InternVLChat):
    """VLMEvalKit wrapper for DPO-tuned InternVL (LoRA adapter)."""
    
    def __init__(
        self,
        base_model_path: str = 'OpenGVLab/InternVL2_5-2B',
        adapter_path: str = None,
        version: str = 'V2.0',
        **kwargs
    ):
        # Don't call parent __init__
        self.use_lmdeploy = False
        self.cot_prompt_version = 'v1'
        self.use_mpo_prompt = False
        self.use_cot = (os.getenv('USE_COT') == '1')
        self.use_postprocess = False
        self.system_prompt = None
        self.cot_prompt = None
        
        self.model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True, use_fast=False
        )
        
        self.pattern = r'Image(\d+)'
        self.replacement = r'Image-\1'
        self.reverse_pattern = r'Image-(\d+)'
        self.reverse_replacement = r'Image\1'
        self.screen_parse = True
        
        # Load base model
        print(f"Loading base model: {base_model_path}")
        self.model = InternVLChatModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Load and merge LoRA adapter
        if adapter_path:
            print(f"Loading DPO adapter: {adapter_path}")
            self.model.language_model = PeftModel.from_pretrained(
                self.model.language_model,
                adapter_path,
            )
            print("Merging adapter weights...")
            self.model.language_model = self.model.language_model.merge_and_unload()
        
        self.model = self.model.cuda().eval()
        self.device = 'cuda'
        
        self.version = version
        self.best_of_n = 1
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
    
    def use_custom_prompt(self, dataset):
        return False



class FinetunedInternVLChat(InternVLChat):
    """Custom wrapper for finetuned InternVL models using InternVL's model class."""
    
    def __init__(self, model_path, version='V2.0', **kwargs):
        from internvl.model.internvl_chat import InternVLChatModel
        
        # Don't call parent __init__ - we replicate it with our custom model loading
        
        # Basic setup from parent
        self.use_lmdeploy = False
        self.cot_prompt_version = 'v1'
        self.use_mpo_prompt = False
        self.use_cot = (os.getenv('USE_COT') == '1')
        self.use_postprocess = False
        self.system_prompt = None
        self.cot_prompt = None
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        # Regex patterns from parent
        self.pattern = r'Image(\d+)'
        self.replacement = r'Image-\1'
        self.reverse_pattern = r'Image-(\d+)'
        self.reverse_replacement = r'Image\1'
        
        self.screen_parse = True
        
        # Load model using InternVL's model class
        self.model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='auto',
        ).eval()
        self.device = 'cuda'
        
        # Rest of parent attributes
        self.version = version
        self.best_of_n = 1
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config.')

    


CUSTOM_MODELS = {
    'InternVL2_5-8B-Refined-v1': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/refined_v1', version='V2.0'),
    'InternVL2_5-8B-Refined-v2': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/refined_v2', version='V2.0'),
    'InternVL2_5-8B-Feedback-v1': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1', version='V2.0'),
    'InternVL2_5-8B-Feedback-Refined-v1-vision-lora_10ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_refined_v1_vision_lora_10ep_8b', version='V2.0'),
    'InternVL2_5-8B-Feedback-Refined-v1-vision-lora_1ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_refined_v1_vision_lora_1ep_8b', version='V2.0'),
    'InternVL2_5-2B-Refined-v1': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/refined_v1_2b', version='V2.0'),
    'InternVL2_5-2B-Refined-v2': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/refined_v2_2b', version='V2.0'),
    'InternVL2_5-2B-Feedback-v1': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1_2b', version='V2.0'),
    'InternVL2_5-2B-Feedback-v1_1ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1_2b_1e', version='V2.0'),
    'InternVL2_5-2B-Feedback-v1_1000ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1_2b_1000e', version='V2.0'),
    'InternVL2_5-2B-Full-v1_10ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/full_v1_2b_10e', version='V2.0'),
    'InternVL2_5-2B-Full-v1_1ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/full_v1_2b_1e', version='V2.0'),
    'InternVL2_5-2B-Feedback-llm-v1_1ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/full_feedback_llm_v1', version='V2.0'),
    'InternVL2_5-2B-Feedback-llm-v1_10ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/full_feedback_llm_v1_10ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-vision-llm-v1_10ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_llm_v1_10ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-vision-llm-v1_1ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_llm_v1_1ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-vision-lora_10ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_10ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-vision-lora_1ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_1ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-vision-lora_100ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_100ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-Refined-v1-vision-lora_10ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_refined_v1_vision_lora_10ep', version='V2.0'),
    'InternVL2_5-2B-Feedback-Refined-v1-vision-lora_100ep': partial(FinetunedInternVLChat, model_path='/home/ubuntu/workspace/elbiat/checkpoints/feedback_refined_v1_vision_lora_100ep', version='V2.0'),
}

# Add to CUSTOM_MODELS dict at the bottom
CUSTOM_MODELS.update({
    'InternVL2_5-2B-DPO-v1': partial(
        DPOInternVLChat,
        base_model_path='OpenGVLab/InternVL2_5-2B',
        adapter_path='/home/ubuntu/workspace/elbiat/checkpoints/dpo_v1/best',
        version='V2.0'
    ),
    'InternVL2_5-2B-DPO-v1-final': partial(
        DPOInternVLChat,
        base_model_path='OpenGVLab/InternVL2_5-2B',
        adapter_path='/home/ubuntu/workspace/elbiat/checkpoints/dpo_v1/final',
        version='V2.0'
    ),
     'InternVL2_5-2B-DPO-v1-ep100': partial(
        DPOInternVLChat,
        base_model_path='OpenGVLab/InternVL2_5-2B',
        adapter_path='/home/ubuntu/workspace/elbiat/checkpoints/dpo_v1_100_ep/best',
        version='V2.0'
    ),
})


def register():
    from vlmeval.config import supported_VLM
    supported_VLM.update(CUSTOM_MODELS)
    print(f"Registered {len(CUSTOM_MODELS)} custom models")