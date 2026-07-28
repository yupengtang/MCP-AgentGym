#!/usr/bin/env bash
# Submit the full model evaluation: base and SFT, each across every split and
# perturbation suite, plus the serving benchmarks.
#
# The SFT jobs depend on the training job, so this can be fired before training
# finishes and Slurm will sequence it.
#
#   bash slurm/run_full_eval.sh [SFT_TRAIN_JOBID]

set -euo pipefail
cd "${AGENTGYM_REPO:-$(dirname "$0")/..}"
source env.sh >/dev/null

TRAIN_JOB="${1:-}"
ADAPTER="${ADAPTER:-$AGENTGYM_ARTIFACTS/sft-qwen3-8b-v2}"
SUITES="${SUITES:-clean rename arg_rename extra_required_arg distractors server_dropout combined}"
SPLITS="${SPLITS:-in_distribution unseen_fault unseen_topology compositional}"
LIMIT="${LIMIT:-40}"

cd slurm/logs

# The base model needs no adapter, so it can start immediately.
BASE=$(sbatch --parsable \
  --export=ALL,VARIANT=base,SUITES="$SUITES",SPLITS="$SPLITS",LIMIT=$LIMIT \
  ../serve_and_eval.sbatch)
echo "base eval        : $BASE"

DEP=""
[ -n "$TRAIN_JOB" ] && DEP="--dependency=afterok:$TRAIN_JOB"

SFT=$(sbatch --parsable $DEP \
  --export=ALL,VARIANT=sft,ADAPTER="$ADAPTER",SUITES="$SUITES",SPLITS="$SPLITS",LIMIT=$LIMIT \
  ../serve_and_eval.sbatch)
echo "sft eval         : $SFT${DEP:+  (after $TRAIN_JOB)}"

BENCH=$(sbatch --parsable $DEP \
  --export=ALL,ADAPTER="$ADAPTER" \
  ../serving_bench.sbatch)
echo "serving benchmark: $BENCH${DEP:+  (after $TRAIN_JOB)}"

echo
echo "when all three finish:  python -m agentgym.eval.report"
