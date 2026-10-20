#!/bin/bash
#SBATCH --job-name=agentgym-egrpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Episode-level GRPO: group-relative advantages over whole incidents.
#
# Phase 1 serves the current policy and rolls out G attempts per incident; phase 2 scores
# each group with the verifiable episode reward and takes advantage-weighted policy
# gradient steps. Both halves share the job because sampling needs the engine and training
# needs the same GPU.
#
# Why this exists alongside dpo: the *same* 2,400 rollouts yield 435 DPO pairs but 2,388
# GRPO episodes across 597 groups, because a preference pair keeps only a best/worst couple
# that clears a margin while a group uses the whole reward distribution.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
module load cuda/12.6.1 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# v4 is the SFT adapter every published row derives from -- verified by DPO-v5's
# saved reference adapter being byte-identical to it. The default used to be v2,
# which is 178% away and produced a silently cross-lineage episode-GRPO result.
ADAPTER="${ADAPTER:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b-v4}"
OUT="${OUT:-$AGENTGYM_ARTIFACTS/episode-grpo-qwen3-8b}"
PORT="${PORT:-8${SLURM_JOB_ID: -3}}"
GROUP="${GROUP:-4}"          # rollouts per incident: the GRPO group size
LIMIT="${LIMIT:-150}"
TEMPERATURE="${TEMPERATURE:-1.0}"   # a group needs reward spread; greedy yields none
SHARDS="${SHARDS:-8}"
SAMPLE_DIR="$AGENTGYM_RUNS/egrpo-samples"

export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
# Long, variable-length episodes fragment the allocator; expandable segments let freed
# blocks be reused across differing sequence lengths instead of stranding them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$SAMPLE_DIR"

# Rollouts can be reused. The DPO sampler already drew 4 attempts per incident from this
# same SFT policy at temperature, which is exactly what phase 1 produces -- so pointing
# SAMPLES_FILE at them makes the first GRPO iteration on-policy without paying for the
# sampling twice. Later iterations must re-sample, because the policy has moved.
if [ -n "${SAMPLES_FILE:-}" ]; then
  [ -s "$SAMPLES_FILE" ] || { echo "SAMPLES_FILE is empty or missing: $SAMPLES_FILE"; exit 1; }
  echo "=== phase 1 skipped: reusing $(wc -l < "$SAMPLES_FILE") episodes from $SAMPLES_FILE ==="
  cp "$SAMPLES_FILE" "$SAMPLE_DIR/samples.jsonl"
else

rm -f "$SAMPLE_DIR"/samples*.jsonl

echo "=== phase 1: roll out $GROUP attempts per incident ==="
"$AGENTGYM_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 \
  --max-model-len 32768 --gpu-memory-utilization 0.88 \
  --chat-template "$(python -m agentgym.train.chat_template --model "$MODEL")" \
  --enable-lora --lora-modules "sft=$ADAPTER" --max-lora-rank 32 \
  --disable-log-requests > "$AGENTGYM_RUNS/vllm-egrpo-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
stop_engine () { kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true; sleep 10; }
trap stop_engine EXIT

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "engine died:"; tail -40 "$AGENTGYM_RUNS/vllm-egrpo-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "engine never ready"; exit 1; }
echo "engine ready"

# Sharded for the same reason DPO sampling is: one process leaves the engine's batch nearly
# empty and caps how much data the run can produce.
pids=()
for s in $(seq 0 $((SHARDS - 1))); do
  python -m agentgym.cli rollout \
    --policy llm --split in_distribution \
    --samples "$GROUP" --limit "$LIMIT" --temperature "$TEMPERATURE" \
    --shard-index "$s" --shard-total "$SHARDS" \
    --gym-offset $((420 + s * 3)) \
    --base-url "http://127.0.0.1:$PORT/v1" --model sft \
    --out "$SAMPLE_DIR/samples-$s.jsonl" --max-steps 24 \
    > "$SAMPLE_DIR/shard-$s.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "  (a sampling shard failed; continuing)"; done
cat "$SAMPLE_DIR"/samples-*.jsonl > "$SAMPLE_DIR/samples.jsonl"
n=$(wc -l < "$SAMPLE_DIR/samples.jsonl")
echo "sampled $n episodes across $SHARDS shards"
[ "$n" -gt 0 ] || { echo "no episodes sampled"; exit 1; }

stop_engine
trap - EXIT

fi

echo "=== phase 2: group-relative advantages + policy gradient ==="
python -m agentgym.train.episode_grpo \
  --trajectories "$SAMPLE_DIR/samples.jsonl" \
  --model "$MODEL" \
  --sft-adapter "$ADAPTER" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-1}" \
  --beta "${BETA:-0.04}" \
  --lr "${LR:-5e-7}"

echo "episode-level GRPO complete -> $OUT"
