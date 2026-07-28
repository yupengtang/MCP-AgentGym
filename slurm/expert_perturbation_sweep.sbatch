#!/bin/bash
#SBATCH --job-name=agentgym-sweep
#SBATCH --partition=ice-cpu
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out

# The scripted expert against every perturbation suite: one array task per suite.
#
# The expert resolves tool names through the perturbed schema rather than from memory,
# so it is the ceiling a model could reach if it read the schema perfectly every time.
# That makes it the reference line the trained variants are measured against, and it
# doubles as an end-to-end check that each perturbation is actually applied.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh

export AGENTGYM_RUNTIME="${TMPDIR:-/tmp}/$USER/agentgym-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$AGENTGYM_RUNTIME"
trap 'rm -rf "$AGENTGYM_RUNTIME"' EXIT

SUITES=(clean rename paraphrase arg_rename extra_required_arg retype_arg \
        distractors server_dropout malformed truncation flaky combined)
SUITE="${SUITES[$SLURM_ARRAY_TASK_ID]}"

echo "suite=$SUITE on $(hostname)"

for split in in_distribution unseen_fault unseen_topology compositional; do
  OUT="$AGENTGYM_RUNS/eval/expert-scripted/$split/$SUITE"
  mkdir -p "$OUT"
  python -m agentgym.cli rollout \
    --policy expert \
    --split "$split" \
    --perturbation "$SUITE" \
    --limit 60 \
    --shard-index 0 --shard-total 1 \
    --gym-offset $((400 + SLURM_ARRAY_TASK_ID * 3)) \
    --out "$OUT/trajectories.jsonl" \
    --max-steps 28
done

echo "suite $SUITE done"
