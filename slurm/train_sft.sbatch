#!/bin/bash
#SBATCH --job-name=agentgym-sft
#SBATCH --partition=gpu
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
export HF_HUB_OFFLINE=1          # base weights are already in the shared cache
export BITSANDBYTES_NOWELCOME=1

MODEL="${MODEL:-${AGENTGYM_BASE_MODEL:-Qwen/Qwen3-8B}}"
# The corpus was hardcoded to $AGENTGYM_DATA/sft. A held-out set other than `published`
# has its own corpus built from its own pool, and training the default one against it
# would put withheld families back into training -- the same silent failure the rollout
# launcher had, and just as invisible: the run completes and only the held-out score is
# wrong, in the flattering direction.
DATASET="${DATASET:-$AGENTGYM_DATA/sft}"
OUT="${OUT:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b}"
# The run name labelled every run "sft-qwen3-8b" no matter what was trained, which is
# harmless while there is one base model and actively misleading once there are two: the
# MLflow run and the second model's adapter would disagree about what produced it.
RUN_NAME="${RUN_NAME:-$(basename "$OUT")}"
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Every knob this launcher accepts, echoed before the run. SEED was exported by three
# separate jobs and forwarded by none of them -- the trainer has --seed and uses it, this
# script simply never passed it, so all three "different seeds" trained at seed 0. The
# resulting 15.6pp gap was published as seed variance and was actually a learning-rate
# difference (5e-5 vs 1e-4). An env var nobody reads is indistinguishable from one that
# does nothing, so the effective configuration is now printed where the log will show it.
echo "--- effective SFT configuration ---"
for kv in "MODEL=$MODEL" "DATASET=$DATASET" "HELDOUT=${AGENTGYM_HELDOUT_SET:-published}" "EPOCHS=${EPOCHS:-3}" "LR=${LR:-1e-4}" "SEED=${SEED:-0}" \
          "GRAD_ACCUM=${GRAD_ACCUM:-8}" "MAX_LEN=${MAX_LEN:-8192}" "LORA_R=${LORA_R:-32}"; do
  echo "  $kv"
done

python -m agentgym.train.sft_train \
  --model "$MODEL" \
  --train-file "$DATASET/sft_train.jsonl" \
  --eval-file  "$DATASET/sft_val.jsonl" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-1e-4}" \
  --seed "${SEED:-0}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --max-length "${MAX_LEN:-8192}" \
  --lora-r "${LORA_R:-32}" \
  --run-name "$RUN_NAME"

echo "SFT complete -> $OUT"
