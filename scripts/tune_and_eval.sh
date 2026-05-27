#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_DIR="${TARGET_DIR}"
SELECTION_RATE="${SELECTION_RATE}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"  # 1e-4 for LLaMA-3-8B, 1e-5 for Qwen2.5 and Mistral-Nemo
USE_LORA="${USE_LORA:-True}"  # 大小写不敏感

if [[ "${TARGET_DIR}" == *random* ]]; then
  MODEL_NAME="${MODEL_NAME:-}"
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "[runner] ERROR: TARGET_DIR contains 'random', but MODEL_NAME is empty. Please export MODEL_NAME before running." >&2
    exit 1
  fi
else
  MODEL_NAME="${TARGET_DIR%%_*}"
fi

echo "[runner] ========== Starting finetune job =========="
echo "[runner] Running finetune command:"
echo "  TARGET_DIR=${TARGET_DIR} SELECTION_RATE=${SELECTION_RATE} LEARNING_RATE=${LEARNING_RATE} USE_LORA=${USE_LORA} MODEL_NAME=${MODEL_NAME} srun --nodes=1 -p H100 --gres=gpu:4 ${SCRIPT_DIR}/finetune.sh"
TARGET_DIR="${TARGET_DIR}" SELECTION_RATE="${SELECTION_RATE}" \
  LEARNING_RATE="${LEARNING_RATE}" USE_LORA="${USE_LORA}" MODEL_NAME="${MODEL_NAME}" \
  srun --nodes=1 -p "H100" --gres=gpu:4 "${SCRIPT_DIR}/finetune.sh"

echo "[runner] ========== Finetune finished, starting evaluation job =========="
if [[ "${TARGET_DIR}" == *random* ]]; then
  ARGS=(
    --target_dir "${TARGET_DIR}"
    --selection_rate "${SELECTION_RATE}"
    --use_lora "${USE_LORA}"
    --base_model_name "${MODEL_NAME}"
  )
else
  ARGS=(
    --target_dir "${TARGET_DIR}"
    --selection_rate "${SELECTION_RATE}"
    --use_lora "${USE_LORA}"
  )
fi

echo "[runner] Running evaluation command:"
echo "  srun --nodes=1 -p H100 --gres=gpu:1 python ${PROJECT_ROOT}/tuning/eval.py ${ARGS[*]}"
srun --nodes=1 -p "H100" --gres=gpu:1 python "${PROJECT_ROOT}/tuning/eval.py" "${ARGS[@]}"

echo "[runner] ========== Evaluation finished =========="
