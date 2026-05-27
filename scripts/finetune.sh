#!/bin/bash
set -euo pipefail

# Resolve project root from script location (works from any cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # /nas/tky/work/Data-Whisperer

# >>>---- Config ---->>>
# Config: launch mode
LAUNCH_MODE="ddp"  # single | ddp
# MASTER_PORT="29501"  # Change this port to avoid conflicts if needed
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT=$((29500 + SLURM_JOB_ID % 100))
else
  MASTER_PORT=29501
fi

# Config: devices
GPUS=4
CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC_PER_NODE="${GPUS}"

# Config: paths
TARGET_DIR="${TARGET_DIR}"
SELECTION_RATE="${SELECTION_RATE}"  # 0.5 means using the top 50% of the pruned data, 1.0 means using all the data
MODEL_NAME="${MODEL_NAME:-}"  # example: Llama-3-8B-Instruct

if [[ "${TARGET_DIR}" == *random* ]]; then
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "[finetune] ERROR: TARGET_DIR contains 'random', but MODEL_NAME is empty. Please export MODEL_NAME before running." >&2
    exit 1
  fi
fi

# Config: train args
NUM_TRAIN_EPOCHS=5
LEARNING_RATE="${LEARNING_RATE:-1e-4}"  # 1e-4 for LLaMA-3-8B, 1e-5 for Qwen2.5 and Mistral-Nemo
PER_DEVICE_TRAIN_BATCH_SIZE=2  # 控制整体batch_size=8
USE_LORA="${USE_LORA:-True}"

# Config: logging
LOGGING_STEPS=100
# <<<---- Config ----<<<

export CUDA_VISIBLE_DEVICES
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH "( sport = :${port} )" | grep -q .
    return
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return
  fi
  echo "[finetune] Warning: neither ss nor lsof is available, skip MASTER_PORT availability check." >&2
  return 1
}

while port_in_use "${MASTER_PORT}"; do
  echo "[finetune] MASTER_PORT=${MASTER_PORT} is in use, trying $((MASTER_PORT + 1))"
  MASTER_PORT="$((MASTER_PORT + 1))"
done

echo "[finetune] Using MASTER_PORT=${MASTER_PORT}"

ARGS=(
  --target_dir "${TARGET_DIR}"
  --model_name "${MODEL_NAME}"
  --selection_rate "${SELECTION_RATE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --use_lora "${USE_LORA}"
  --logging_steps "${LOGGING_STEPS}"
)

echo "[finetune] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[finetune] LAUNCH_MODE=${LAUNCH_MODE}, GPUS=${GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[finetune] TARGET_DIR=${TARGET_DIR}"
echo "[finetune] SELECTION_RATE=${SELECTION_RATE}"
echo "[finetune] LEARNING_RATE=${LEARNING_RATE}"
echo "[finetune] USE_LORA=${USE_LORA}"

# Examples:
# single: LAUNCH_MODE=single GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/finetune.sh
# ddp:    LAUNCH_MODE=ddp GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/finetune.sh

if [[ "${LAUNCH_MODE}" == "single" ]]; then
  CMD=(python "${PROJECT_ROOT}/tuning/tuner.py" "${ARGS[@]}")
elif [[ "${LAUNCH_MODE}" == "ddp" ]]; then
  CMD=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=${MASTER_PORT} "${PROJECT_ROOT}/tuning/tuner.py" "${ARGS[@]}")
else
  echo "[finetune] Unsupported LAUNCH_MODE=${LAUNCH_MODE}. Use single|ddp" >&2
  exit 1
fi

echo -e "[finetune] Command:\n  ${CMD[*]}"
echo "---------- start tuning ----------"

# run tuner.py
"${CMD[@]}"  # > "${OUTPUT_DIR}/info.log" 2>&1
