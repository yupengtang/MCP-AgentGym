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

# Every knob this launcher accepts, echoed before the run. SEED was exported by three
# separate jobs and forwarded by none of them -- the trainer has --seed and uses it, this
# script simply never passed it, so all three "different seeds" trained at seed 0. The
# resulting 15.6pp gap was published as seed variance and was actually a learning-rate
# difference (5e-5 vs 1e-4). An env var nobody reads is indistinguishable from one that
# does nothing, so the effective configuration is now printed where the log will show it.
echo "--- effective SFT configuration ---"
for kv in "EPOCHS=${EPOCHS:-3}" "LR=${LR:-1e-4}" "SEED=${SEED:-0}" \
          "GRAD_ACCUM=${GRAD_ACCUM:-8}" "MAX_LEN=${MAX_LEN:-8192}" "LORA_R=${LORA_R:-32}"; do
  echo "  $kv"
done

python -m agentgym.train.sft_train \
  --train-file "$AGENTGYM_DATA/sft/sft_train.jsonl" \
  --eval-file  "$AGENTGYM_DATA/sft/sft_val.jsonl" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-1e-4}" \
  --seed "${SEED:-0}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --max-length "${MAX_LEN:-8192}" \
  --lora-r "${LORA_R:-32}" \
  --run-name "sft-qwen3-8b"

echo "SFT complete -> $OUT"
