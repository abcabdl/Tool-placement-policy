#!/usr/bin/env bash
set -euo pipefail

# Stage 1: supervised agent adaptation without vocabulary expansion.
# Edit these paths before running.
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/path/to/agent_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage1_lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_DIR}/runs"

python -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  -m swift.cli.sft \
  --train_type lora \
  --lora_target_modules ALL \
  --lora_rank 64 \
  --lora_alpha 128 \
  --torch_dtype bfloat16 \
  --model "${MODEL_PATH}" \
  --model_type qwen2 \
  --template qwen2_5 \
  --dataset "${DATASET_PATH}" \
  --max_length 32768 \
  --learning_rate 1e-4 \
  --num_train_epochs 4.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
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
