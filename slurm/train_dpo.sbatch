#!/bin/bash
#SBATCH --job-name=agentgym-dpo
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Mine preference pairs from the SFT policy, then run DPO on them.
#
# Both halves live in one job because the sampling phase needs the SFT model served and
# the training phase needs the same GPU: serving first, then tearing the engine down and
# training on the pairs it produced.
#
# Pairs are drawn from repeated attempts at the *same* incident, so both sides of a
# preference faced an identical cluster and the contrast is behaviour, not difficulty.
# Sampling is at temperature -- a deterministic policy yields identical attempts and
# therefore zero usable pairs.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
module load cuda/12.6.1 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ADAPTER="${ADAPTER:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b-v2}"
OUT="${OUT:-$AGENTGYM_ARTIFACTS/dpo-qwen3-8b}"
PORT="${PORT:-8${SLURM_JOB_ID: -3}}"
SAMPLES="${SAMPLES:-4}"
LIMIT="${LIMIT:-120}"
TEMPERATURE="${TEMPERATURE:-0.9}"
# Per-run output paths. These were hard-coded, which meant a second DPO run silently
# overwrote the rollouts and preference pairs backing the published DPO-435 result -- and
# the same rollouts episode-GRPO trained on. Nothing would have errored; the files would
# simply have been different afterwards.
SAMPLE_DIR="${SAMPLE_DIR:-$AGENTGYM_RUNS/dpo-samples}"
PAIRS="${PAIRS:-$AGENTGYM_DATA/sft/dpo_pairs.jsonl}"

export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$OUT" "$SAMPLE_DIR"

# A run that overwrites its inputs is unrecoverable and produces no error, so it has to be
# opted into explicitly rather than avoided by remembering to pass SAMPLE_DIR.
if [ -z "${OVERWRITE:-}" ] && compgen -G "$SAMPLE_DIR/samples*.jsonl" > /dev/null; then
  echo "REFUSING TO RUN: $SAMPLE_DIR already holds samples ($(cat "$SAMPLE_DIR"/samples-*.jsonl 2>/dev/null | wc -l) episodes)." >&2
  echo "  Pass SAMPLE_DIR=<new dir> for an independent run, or OVERWRITE=1 to replace them." >&2
  exit 1
fi
if [ -z "${OVERWRITE:-}" ] && [ -s "$PAIRS" ]; then
  echo "REFUSING TO RUN: $PAIRS already exists ($(wc -l < "$PAIRS") pairs)." >&2
  echo "  Pass PAIRS=<new path> for an independent run, or OVERWRITE=1 to replace it." >&2
  exit 1
fi

echo "=== phase 1: serve the SFT policy and sample $SAMPLES attempts per incident ==="
"$AGENTGYM_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 \
  --max-model-len 32768 --gpu-memory-utilization 0.88 \
  --chat-template src/agentgym/train/agentgym_chat_template.jinja \
  --enable-lora --lora-modules "sft=$ADAPTER" --max-lora-rank 32 \
  --disable-log-requests > "$AGENTGYM_RUNS/vllm-dpo-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
stop_engine () { kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true; sleep 10; }
trap stop_engine EXIT

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "engine died:"; tail -40 "$AGENTGYM_RUNS/vllm-dpo-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "engine never ready"; exit 1; }
echo "engine ready"

# Sampling is latency-bound on the model, not compute-bound on the environment. Running
# it in one process leaves the engine's batch nearly empty and caps how many preference
# pairs the run can possibly produce -- which was the real reason the first DPO attempt
# had only 79 pairs to learn from. Fan out shards against the same engine.
SHARDS="${SHARDS:-8}"
pids=()
for s in $(seq 0 $((SHARDS - 1))); do
  python -m agentgym.cli rollout \
    --policy llm --split in_distribution \
    --samples "$SAMPLES" --limit "$LIMIT" \
    --temperature "$TEMPERATURE" \
    --shard-index "$s" --shard-total "$SHARDS" \
    --gym-offset $((320 + s * 3)) \
    --base-url "http://127.0.0.1:$PORT/v1" --model sft \
    --out "$SAMPLE_DIR/samples-$s.jsonl" --max-steps 24 \
    > "$SAMPLE_DIR/shard-$s.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "  (a sampling shard failed; continuing)"; done
cat "$SAMPLE_DIR"/samples-*.jsonl > "$SAMPLE_DIR/samples.jsonl"
echo "sampled $(wc -l < "$SAMPLE_DIR/samples.jsonl") episodes across $SHARDS shards"

stop_engine
trap - EXIT

echo "=== phase 2: mine pairs ==="
export PAIRS SAMPLE_DIR
python - <<'PY'
import json, os
from pathlib import Path
from agentgym import config
from agentgym.data import curate
from agentgym.data import trajectory as T

# Read from the same directory phase 1 wrote to. Deriving this independently meant a
# run with a custom SAMPLE_DIR sampled into one place and mined pairs from another.
samples = Path(os.environ["SAMPLE_DIR"]) / "samples.jsonl"
trajectories = T.read_jsonl(samples)
eval_ids = set(json.loads((config.DATA_DIR / "sft" / "eval_incident_ids.json").read_text()))
pairs, stats = curate.build_dpo(trajectories, eval_incident_ids=eval_ids)
out = Path(os.environ["PAIRS"])
curate.write_jsonl(pairs, out)
print(json.dumps(stats, indent=2))
if not pairs:
    raise SystemExit("no preference pairs mined; the policy may be too deterministic "
                     "(raise --temperature) or too uniformly good/bad")
PY

echo "=== phase 3: DPO ==="
python -m agentgym.train.dpo_train \
  --pairs-file "$PAIRS" \
  --sft-adapter "$ADAPTER" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-1}" \
  --beta "${BETA:-0.1}" \
  --run-name dpo-qwen3-8b

echo "DPO complete -> $OUT"
