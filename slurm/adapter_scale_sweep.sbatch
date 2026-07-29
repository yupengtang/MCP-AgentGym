#!/bin/bash
#SBATCH --job-name=agentgym-scale
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Trace the specialization/generalization frontier with one knob.
#
# Every post-trained variant scores below the base model on held-out fault families, and
# the measured mechanism is label-prior collapse: SFT/DPO emit the two never-demonstrated
# diagnosis labels 0 times in ~600 submissions, against 5.0% for base. The labels remain in
# the tool enum -- the policy simply stopped being able to reach them.
#
# Scaling the LoRA update by s interpolates between base (s=0) and the trained policy
# (s=1) without retraining, so one sweep shows whether the in-distribution gain and the
# held-out regression can be separated, and where the best operating point is.
#
# All four adapters are served by a single engine via multi-LoRA, so the sweep costs one
# model load rather than four.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
module load cuda/12.6.1 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-8B}"
FULL="${FULL:-$AGENTGYM_ARTIFACTS/dpo-qwen3-8b-v5}"
SCALED="${SCALED:-$AGENTGYM_ARTIFACTS/scaled}"
PORT="${PORT:-8${SLURM_JOB_ID: -3}}"
LIMIT="${LIMIT:-64}"
SPLITS="${SPLITS:-in_distribution unseen_fault}"
# Separate OUT per sweep: mixing cells of different --limit into one directory
# would let scale_report average populations of different size and composition.
OUT="${OUT:-$AGENTGYM_RUNS/eval-scale}"
TEMPLATE="src/agentgym/train/agentgym_chat_template.jinja"

export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$OUT"

# name=path pairs served as LoRA modules. Overridable so the same sweep can attenuate a
# different adapter family -- the mechanism under test is closed-catalogue fine-tuning, not
# one checkpoint, so it has to be runnable against more than one.
MODULES="${MODULES:-dpo5s025=$SCALED/dpo5-s025 dpo5s05=$SCALED/dpo5-s05 dpo5s075=$SCALED/dpo5-s075 dpo5s10=$FULL}"
for spec in $MODULES; do
  d="${spec#*=}"
  [ -f "$d/adapter_model.safetensors" ] || { echo "missing adapter: $d"; exit 1; }
done

echo "=== serving base + adapter scales on one engine ==="
echo "    modules: $MODULES"
"$AGENTGYM_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 \
  --max-model-len 32768 --gpu-memory-utilization 0.88 \
  --chat-template "$TEMPLATE" \
  --enable-lora --max-lora-rank 32 --max-loras 4 \
  --lora-modules $MODULES \
  --disable-log-requests > "$AGENTGYM_RUNS/vllm-scale-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "engine died:"; tail -40 "$AGENTGYM_RUNS/vllm-scale-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "engine never ready"; exit 1; }
echo "engine ready"

# s=0 is the base model itself, served with no adapter -- it anchors the low end of the
# sweep and is the variant the regression is measured against. The variant label on each
# trajectory is derived from the served model name, so the names above are chosen to make
# `_variant_of` produce one distinct row per scale.
# Rollout is latency-bound on the model, not compute-bound on the environment, so a single
# process leaves the engine's batch nearly empty. Measured unsharded: 3 episodes in 18
# minutes, which puts this sweep at ~64 hours. This is the same defect as the DPO sampler's
# `--shard-total 1` (reports/EXPERIMENTS.md) -- it never errors, it is only slow, which is
# why it is easy to write a second time.
SHARDS="${SHARDS:-8}"
FAILED=0
VARIANTS="${VARIANTS:-$MODEL dpo5s025 dpo5s05 dpo5s075 dpo5s10}"
for SERVED in $VARIANTS; do
  VARIANT="$SERVED"; [ "$SERVED" = "$MODEL" ] && VARIANT="base"
  for split in $SPLITS; do
    dest="$OUT/$VARIANT-$split-clean.jsonl"
    echo "--- $VARIANT / $split ($SHARDS shards) ---"
    rm -f "$OUT/.part-$VARIANT-$split-"*.jsonl
    pids=()
    for s in $(seq 0 $((SHARDS - 1))); do
      python -m agentgym.cli rollout \
        --policy llm --split "$split" --perturbation clean \
        --limit "$LIMIT" --temperature 0.0 --max-steps 24 \
        --shard-index "$s" --shard-total "$SHARDS" \
        --gym-offset $((520 + s * 3)) \
        --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED" \
        --out "$OUT/.part-$VARIANT-$split-$s.jsonl" \
        > "$OUT/log-$VARIANT-$split-$s.txt" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "  (a shard failed; continuing)"; done
    cat "$OUT/.part-$VARIANT-$split-"*.jsonl > "$dest" 2>/dev/null || true
    echo "  -> $(wc -l < "$dest" 2>/dev/null || echo 0) episodes"
    # An empty cell must fail the job. A silently missing cell once produced a complete
    # -looking report built entirely from stale files.
    if [ ! -s "$dest" ]; then echo "EMPTY CELL: $VARIANT/$split"; FAILED=1; fi
  done
done

[ "$FAILED" -eq 0 ] || { echo "one or more cells were empty"; exit 1; }
echo "adapter scale sweep complete -> $OUT"
