# scripts/finetune.sh
#!/bin/bash
set -e

# Defaults
GPUS=${GPUS:-1}
BASE_MODEL=${BASE_MODEL:-"OpenGVLab/InternVL2_5-8B"}
TRAIN_DATA=${TRAIN_DATA:-"feedback_data/refined_v1/train.jsonl"}  # Can be comma-separated
OUTPUT_NAME=${OUTPUT_NAME:-"refine_v1"}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
LORA_RANK=${LORA_RANK:-16}

# Paths
ELBIAT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERNVL_ROOT="${ELBIAT_ROOT}/external/InternVL/internvl_chat"
OUTPUT_DIR="${ELBIAT_ROOT}/checkpoints/${OUTPUT_NAME}"

echo "=== Elbiat Finetuning ==="
echo "Base model: ${BASE_MODEL}"
echo "Train data: ${TRAIN_DATA}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS}"

# Create meta file for InternVL
META_FILE="${INTERNVL_ROOT}/shell/data/elbiat_finetune.json"

# Build JSON with multiple datasets
python3 << EOF
import json
import os

train_data_str = "${TRAIN_DATA}"
elbiat_root = "${ELBIAT_ROOT}"

# Split by comma for multiple files
train_files = [f.strip() for f in train_data_str.split(',')]

meta = {}
for i, train_file in enumerate(train_files):
    # Use absolute path for annotation
    if not os.path.isabs(train_file):
        full_path = os.path.join(elbiat_root, train_file)
    else:
        full_path = train_file
    
    # Check file exists
    if not os.path.exists(full_path):
        print(f"ERROR: File not found: {full_path}")
        exit(1)
    
    # Count lines
    with open(full_path, 'r') as f:
        length = sum(1 for _ in f)
    
    # Create dataset entry
    dataset_name = f"elbiat_train_{i}" if len(train_files) > 1 else "elbiat_train"
    meta[dataset_name] = {
        "root": elbiat_root,  # Root for image paths
        "annotation": full_path,  # Absolute path for annotation file
        "data_augment": False,
        "repeat_time": 1,
        "length": length
    }

with open("${META_FILE}", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Created meta file with {len(meta)} dataset(s):")
for name, info in meta.items():
    print(f"  {name}: {info['length']} examples")
    print(f"    root: {info['root']}")
    print(f"    annotation: {info['annotation']}")
EOF

# Run finetuning
cd "${INTERNVL_ROOT}"
export PYTHONPATH="${INTERNVL_ROOT}:${PYTHONPATH}"

GRADIENT_ACC=$((16 / BATCH_SIZE / GPUS))

export LAUNCHER=pytorch

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=34229 \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${BASE_MODEL}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_FILE}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora ${LORA_RANK} \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version "v2" \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard"

echo "=== Finetuning complete ==="
echo "Model saved to: ${OUTPUT_DIR}"