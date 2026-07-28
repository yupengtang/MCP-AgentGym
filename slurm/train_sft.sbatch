#!/bin/bash
#SBATCH --job-name=agentgym-sft
#SBATCH --partition=ice-gpu
# Explicit GPU model: the partition also contains 16GB V100s, which cannot hold an 8B
# model at 8k context even in 4-bit.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh

module load cuda/12.6.1 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # Qwen3-8B is already in the shared cache
export BITSANDBYTES_NOWELCOME=1

OUT="${OUT:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b}"
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m agentgym.train.sft_train \
  --train-file "$AGENTGYM_DATA/sft/sft_train.jsonl" \
  --eval-file  "$AGENTGYM_DATA/sft/sft_val.jsonl" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-1e-4}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --max-length "${MAX_LEN:-8192}" \
  --lora-r "${LORA_R:-32}" \
  --run-name "sft-qwen3-8b"

echo "SFT complete -> $OUT"
