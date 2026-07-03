#!/usr/bin/env bash
set -euo pipefail

# Simple single-GPU evaluation template based on SWIFT Native + vLLM.
# Edit environment variables or pass them inline, for example:
# MODEL_PATH=/path/to/model EVAL_DATASET=tau2_bench GPU_ID=0 bash examples/evaluate_swift_native.sh

MODEL_PATH="${MODEL_PATH:-/path/to/AgentVocab-model}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
MODEL_TYPE="${MODEL_TYPE:-qwen2}"
TEMPLATE="${TEMPLATE:-qwen2_5}"
EVAL_DATASET="${EVAL_DATASET:-tau2_bench}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/${EVAL_DATASET}/${MODEL_NAME}}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10000}"
MAX_LENGTH="${MAX_LENGTH:-30000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

DATASET_ARGS=()
if [[ "${EVAL_DATASET}" == "tau_bench" || "${EVAL_DATASET}" == "tau2_bench" ]]; then
  if [[ -z "${TAU_BENCH_API_KEY:-}" ]]; then
    echo "TAU_BENCH_API_KEY is required for ${EVAL_DATASET}." >&2
    exit 1
  fi
  DATASET_ARGS=(
    --eval_dataset_args "{\"${EVAL_DATASET}\": {\"extra_params\": {\"api_key\": \"${TAU_BENCH_API_KEY}\"}}}"
  )
fi

swift eval \
  --log_level info \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --template "${TEMPLATE}" \
  --eval_backend Native \
  --infer_backend vllm \
  --port "${PORT}" \
  --eval_dataset "${EVAL_DATASET}" \
  "${DATASET_ARGS[@]}" \
  --max_length "${MAX_LENGTH}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --vllm_max_model_len "${MAX_MODEL_LEN}" \
  --vllm_enable_prefix_caching true \
  --truncation_strategy left \
  --eval_generation_config "{\"max_tokens\": ${MAX_TOKENS}}" \
  2>&1 | tee "${OUTPUT_DIR}/eval_${MODEL_NAME}.log"

