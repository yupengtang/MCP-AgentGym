#!/usr/bin/env bash
# One post-training cycle: curate -> train -> evaluate -> report.
#
# Run this after an expert corpus has been generated. Slurm dependencies chain the GPU
# stages, so the whole cycle can be fired once and left alone.
#
#   bash slurm/run_sft_cycle.sh [TRAJECTORY_DIR] [TAG]

set -euo pipefail
cd "${AGENTGYM_REPO:-$(dirname "$0")/..}"
source env.sh >/dev/null

TRAJ="${1:-$AGENTGYM_RUNS/expert-large}"
TAG="${2:-v4}"
ADAPTER="$AGENTGYM_ARTIFACTS/sft-qwen3-8b-$TAG"
SUITES="${SUITES:-clean rename arg_rename extra_required_arg distractors server_dropout combined}"
SPLITS="${SPLITS:-in_distribution unseen_fault unseen_topology compositional}"
LIMIT="${LIMIT:-64}"       # total episodes per cell, split across shards
SHARDS="${SHARDS:-8}"

echo "=== curating datasets from $TRAJ ==="
python -m agentgym.cli build-datasets --trajectories "$TRAJ"
TRAIN_N=$(wc -l < "$AGENTGYM_DATA/sft/sft_train.jsonl")
echo "training examples: $TRAIN_N"
if [ "$TRAIN_N" -lt 200 ]; then
  echo "refusing to train on $TRAIN_N examples; generate more trajectories first" >&2
  exit 1
fi

cd slurm/logs
SFT=$(sbatch --parsable --export=ALL,OUT="$ADAPTER",EPOCHS="${EPOCHS:-2}",LR="${LR:-5e-5}" \
        ../train_sft.sbatch)
echo "sft train        : $SFT  -> $ADAPTER"

BASE=$(sbatch --parsable \
  --export=ALL,VARIANT=base,SUITES="$SUITES",SPLITS="$SPLITS",LIMIT=$LIMIT,SHARDS=$SHARDS \
  ../serve_and_eval.sbatch)
echo "base eval        : $BASE"

EVAL=$(sbatch --parsable --dependency=afterok:$SFT \
  --export=ALL,VARIANT=sft,ADAPTER="$ADAPTER",SUITES="$SUITES",SPLITS="$SPLITS",LIMIT=$LIMIT,SHARDS=$SHARDS \
  ../serve_and_eval.sbatch)
echo "sft eval         : $EVAL  (after $SFT)"

DPO=$(sbatch --parsable --dependency=afterok:$SFT \
  --export=ALL,ADAPTER="$ADAPTER",OUT="$AGENTGYM_ARTIFACTS/dpo-qwen3-8b-$TAG" \
  ../train_dpo.sbatch)
echo "dpo              : $DPO  (after $SFT)"

echo
echo "when finished:  python -m agentgym.eval.report"
