#!/usr/bin/env bash
set -euo pipefail

# Stage 2: continue LoRA SFT after vocabulary expansion.
# BASE_MODEL should point to the output of scripts/expand_tokenizer.py.
BASE_MODEL="${BASE_MODEL:-outputs/expanded_model_step0}"
DATASET_PATH="${DATASET_PATH:-/path/to/agent_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_DIR}/runs"

python -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  -m swift.cli.sft \
  --train_type lora \
  --target_modules all-linear \
  --modules_to_save embed_tokens lm_head \
  --lora_rank 64 \
  --lora_alpha 128 \
  --torch_dtype bfloat16 \
  --model "${BASE_MODEL}" \
  --model_type qwen2 \
  --template qwen2_5 \
  --dataset "${DATASET_PATH}" \
  --max_length 8192 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --num_train_epochs 3.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_steps 500 \
  --save_steps 500 \
  --attn_impl flash_attn \
  --agent_template hermes \
  --lazy_tokenize true \
  --dataloader_num_workers 8 \
  --gradient_checkpointing true \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${OUTPUT_DIR}/runs" \
  --add_version false
