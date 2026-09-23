#!/bin/bash
#SBATCH --job-name=agentgym-bench
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Serving benchmarks for multi-turn agent workloads.
#
# Run twice: once with prefix caching on (vLLM's default) and once with it disabled, so
# the measurement isolates what the shared agent prefix is actually worth rather than
# reporting an unattributed throughput number.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
module load cuda/12.6.1 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ADAPTER="${ADAPTER:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b}"
PORT="${PORT:-8${SLURM_JOB_ID: -3}}"
OUTDIR="$AGENTGYM_ARTIFACTS/serving"
mkdir -p "$OUTDIR"

export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

run_engine () {
  local label="$1"; shift
  echo "=== engine: $label ==="
  "$AGENTGYM_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    --port "$PORT" --host 127.0.0.1 \
    --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --chat-template "$(python -m agentgym.train.chat_template --model "$MODEL")" \
    --disable-log-requests "$@" > "$AGENTGYM_RUNS/vllm-$label-$SLURM_JOB_ID.log" 2>&1 &
  ENGINE_PID=$!
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 $ENGINE_PID 2>/dev/null || { echo "engine died:"; tail -30 "$AGENTGYM_RUNS/vllm-$label-$SLURM_JOB_ID.log"; return 1; }
    sleep 10
  done
  echo "engine never ready"; return 1
}

stop_engine () { kill ${ENGINE_PID:-0} 2>/dev/null || true; wait ${ENGINE_PID:-0} 2>/dev/null || true; sleep 8; }
trap stop_engine EXIT

# 1. Prefix caching ON, with the SFT adapter loaded for the multi-LoRA measurement.
if run_engine "cached" --enable-prefix-caching --enable-lora \
      --lora-modules "sft=$ADAPTER" --max-lora-rank 32; then
  python -m agentgym.eval.serving_bench \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --backend-name "vllm+prefix-cache" --adapters sft \
    --out "$OUTDIR/vllm_prefix_cache_on.json"
fi
stop_engine

# 2. Prefix caching OFF: the control that makes the comparison meaningful.
if run_engine "nocache" --no-enable-prefix-caching; then
  python -m agentgym.eval.serving_bench \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --backend-name "vllm-no-prefix-cache" --skip structured lora \
    --out "$OUTDIR/vllm_prefix_cache_off.json"
fi
stop_engine

echo "serving benchmarks complete -> $OUTDIR"
