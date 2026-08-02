#!/bin/bash
#SBATCH --job-name=agentgym-actgrpo
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Action-level GRPO: one tool call at a time, against the verifiable action reward.
#
# This trainer existed for weeks without a launcher, which is why it was the one component
# the README claimed and never ran. The episode-level variant in train_episode_grpo.sbatch
# credits a whole incident; this one optimises a single call, and the two answer different
# questions -- so both belong in the table rather than one standing in for the other.
#
# Reward per completion (all decided by a parser, a JSON Schema, or the safety policy;
# never by a model):
#   +1.0 parses  +1.0 known tool  +1.0 schema-valid  +0.5 matches expert  -3.5 unsafe
#
# The -3.5 is sized so a well-formed forbidden action scores below emitting nothing
# parseable (-0.5 against 0.0), which tests/test_reward.py asserts.
#
# Memory, decided by arithmetic after three wrong diagnoses.
#
# TRL's GRPO materialises full-vocab logits and `_cast_lm_head_to_fp32` makes them fp32, so
# one tensor is gen x tokens x 151,936 x 4 bytes and the objective holds roughly three
# (policy, old policy, reference-via-disabled-adapter). Batch size barely matters; the
# product that matters is generations x sequence length:
#
#   gen 4, cap 6144 -> 15.6 GB each, ~54.7 GB total   OOM (this is what failed three times)
#   gen 4, cap 4096 -> 10.6 GB each, ~39.7 GB total   fits, keeps 48.7% of prompts
#   gen 2, cap 6144 ->  7.8 GB each, ~31.3 GB total   fits, keeps 85%, but a group of 2
#
# Reducing batch 8 -> 2 -> 1 changed nothing, because batch was never the term. The
# identical "tried to allocate 4.08 GiB" in all three logs was the clue.
#
# Taking gen 4 / cap 4096. A group of 2 standardises to +-1 and is a preference pair
# wearing GRPO's name, which would make the comparison against DPO meaningless. The cost is
# real and is recorded rather than hidden: length correlates with turn depth, so a 4096 cap
# biases training toward early-episode states. That bias is now forced by hardware, not
# chosen -- and it is why this run is reported as a constrained variant of action-level
# GRPO rather than as the method evaluated cleanly.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
module load cuda/12.6.1 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# v4 is the base every published row derives from; see train_dpo.sbatch for the evidence.
ADAPTER="${ADAPTER:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b-v4}"
OUT="${OUT:-$AGENTGYM_ARTIFACTS/grpo-action-v1}"
# Prompts come from verified trajectories: each step's history is a prompt, and the next
# tool call is what the group samples over.
TRAJ="${TRAJ:-$AGENTGYM_RUNS/dpo-samples/samples.jsonl}"

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

[ -s "$TRAJ" ] || { echo "no trajectories at $TRAJ"; exit 1; }
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "missing adapter: $ADAPTER"; exit 1; }
echo "base adapter: $ADAPTER"
echo "prompts from: $TRAJ ($(wc -l < "$TRAJ") episodes)"

python -m agentgym.train.grpo_train \
  --trajectories "$TRAJ" \
  --model "$MODEL" \
  --sft-adapter "$ADAPTER" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-1e-6}" \
  --num-generations "${NUM_GEN:-4}" \
  --batch-size "${BATCH:-1}" \
  --grad-accum "${GRAD_ACCUM:-32}" \
  --max-prompt-length "${MAX_PROMPT:-4096}" \
  --max-completion-length "${MAX_COMPLETION:-256}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --beta "${BETA:-0.04}" \
  --max-prompts "${MAX_PROMPTS:-2000}" \
  --seed "${SEED:-0}"

echo "action-level GRPO complete -> $OUT"
