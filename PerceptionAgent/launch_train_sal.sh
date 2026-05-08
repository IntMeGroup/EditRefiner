#!/usr/bin/env bash
set -euo pipefail

# ?? 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ?? 可选：加载配置（如果没有也不报错）
if [[ -f "config.env" ]]; then
  source "config.env"
else
  echo "config.env not found, using defaults"
fi

# ?? 默认 GPU（如果 config.env 没写）
GPU_IDS="${GPU_IDS:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "GPU_IDS is empty"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=8

# ? 用当前环境 python（关键修改）
PYTHON_BIN=$(which python)
echo "Using python: $PYTHON_BIN"

# ? 启动 DDP
nohup $PYTHON_BIN -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port=29600 \
  train_qwen3_vl_sal.py \
  --seed "${SEED:-42}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --max_length "${MAX_LENGTH:-512}" \
  --max_pixels_per_image "${MAX_PIXELS_PER_IMAGE:-262144}" \
  --precision "${PRECISION:-bf16}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,v_proj}" \
  > train_a.log 2>&1 &