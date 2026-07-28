#!/bin/bash
#SBATCH --job-name=agentgym-eval
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Serve a model with vLLM, then drive the gym against it.
#
# Serving and rollout share one job because the environment is CPU-bound and the engine
# is GPU-bound: running them on the same node keeps the GPU busy while episodes step the
# simulated cluster, and avoids exposing the endpoint across the cluster network.
#
# Env:
#   ADAPTER   path to a LoRA adapter (optional)
#   VARIANT   label recorded on every trajectory (base | sft | dpo | grpo)
#   SUITES    space-separated perturbation suites
#   SPLITS    space-separated incident splits

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
module load cuda/12.6.1 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-8B}"
VARIANT="${VARIANT:-base}"
ADAPTER="${ADAPTER:-}"
SUITES="${SUITES:-clean rename combined server_dropout}"
SPLITS="${SPLITS:-in_distribution unseen_fault unseen_topology compositional}"
PORT="${PORT:-8${SLURM_JOB_ID: -3}}"
LIMIT="${LIMIT:-60}"
TEMPLATE="src/agentgym/train/agentgym_chat_template.jinja"

export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

SERVE_ARGS=(
  --model "$MODEL"
  --served-model-name "$MODEL"
  --port "$PORT" --host 127.0.0.1
  --max-model-len 32768
  --gpu-memory-utilization 0.88
  --chat-template "$TEMPLATE"
  --disable-log-requests
)
if [ -n "$ADAPTER" ]; then
  # Multi-LoRA: base and adapter are both addressable by name on one engine.
  SERVE_ARGS+=(--enable-lora --lora-modules "$VARIANT=$ADAPTER" --max-lora-rank 32)
fi

echo "starting vLLM on :$PORT (model=$MODEL adapter=${ADAPTER:-none})"
"$AGENTGYM_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server "${SERVE_ARGS[@]}" \
  > "$AGENTGYM_RUNS/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "waiting for engine..."
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "engine ready after ${i}0s"; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM died during startup:"; tail -40 "$AGENTGYM_RUNS/vllm-$SLURM_JOB_ID.log"; exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "engine never became ready"; exit 1; }

# The model name the OpenAI API should target: the adapter if there is one, else base.
TARGET="$MODEL"
[ -n "$ADAPTER" ] && TARGET="$VARIANT"

SHARDS="${SHARDS:-8}"

run_cell () {
  local split="$1" suite="$2"
  local out="$AGENTGYM_RUNS/eval/$VARIANT/$split/$suite"
  mkdir -p "$out"
  echo "=== $VARIANT | split=$split | suite=$suite (${SHARDS} shards) ==="

  # Episodes are latency-bound on the model, not compute-bound on the environment: a
  # single sequential worker leaves the GPU idle between turns and costs ~255s per
  # episode. Fanning out shards keeps the engine's batch full and turns a 28-hour sweep
  # into a few hours. Each shard owns its own control plane (disjoint port block via
  # gym-offset) and its own slice of incidents, so they never interact.
  local pids=()
  for s in $(seq 0 $((SHARDS - 1))); do
    python -m agentgym.cli rollout \
      --policy llm \
      --split "$split" \
      --perturbation "$suite" \
      --limit "$LIMIT" \
      --shard-index "$s" --shard-total "$SHARDS" \
      --gym-offset $((300 + s * 3)) \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model "$TARGET" \
      --out "$out/trajectories-$s.jsonl" \
      --max-steps 20 > "$out/shard-$s.log" 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  [ "$failed" -gt 0 ] && echo "  ($failed/$SHARDS shards failed)"

  local n
  n=$(cat "$out"/trajectories-*.jsonl 2>/dev/null | wc -l)
  echo "  episodes: $n"

  # A cell that produced nothing must fail the job. Otherwise the batch run prints
  # "evaluation complete", the report reads whatever stale files happen to be on disk,
  # and the result looks entirely normal. That is how a broken CLI import once produced
  # a plausible full results table from a job that ran for three minutes.
  if [ "$n" -eq 0 ]; then
    echo "  FATAL: cell produced 0 episodes; first shard log follows" >&2
    tail -25 "$out/shard-0.log" >&2 2>/dev/null || true
    EMPTY_CELLS=$((EMPTY_CELLS + 1))
  fi
}
EMPTY_CELLS=0

# The two reported tables need different slices, not their cross-product:
#
#   generalization  = clean tools across every split
#   robustness      = every perturbation suite on one split
#
# Running the full 7x4 grid would cost ~2.8x more GPU for cells neither table reads.
# in_distribution is the right split to perturb: it isolates the effect of the tool
# surface changing, rather than confounding it with an unfamiliar fault or topology.
for split in $SPLITS; do
  run_cell "$split" clean
done

for suite in $SUITES; do
  [ "$suite" = "clean" ] && continue      # already covered above
  run_cell in_distribution "$suite"
done

if [ "${EMPTY_CELLS:-0}" -gt 0 ]; then
  echo "evaluation FAILED for variant=$VARIANT: $EMPTY_CELLS cell(s) produced no episodes" >&2
  exit 1
fi
echo "evaluation complete for variant=$VARIANT"
