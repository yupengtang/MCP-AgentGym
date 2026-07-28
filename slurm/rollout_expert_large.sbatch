#!/bin/bash
#SBATCH --job-name=agentgym-expert-lg
#SBATCH --partition=ice-cpu
#SBATCH --array=0-23
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out

# Larger expert corpus. 614 curated demonstrations proved too thin: the expert follows a
# fixed investigation order per fault family, so a small sample teaches the phrasing
# without teaching the behaviour. This widens the incident pool 3x.
set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh
export AGENTGYM_RUNTIME="${TMPDIR:-/tmp}/$USER/agentgym-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$AGENTGYM_RUNTIME"; trap 'rm -rf "$AGENTGYM_RUNTIME"' EXIT

OUT="$AGENTGYM_RUNS/expert-large"; mkdir -p "$OUT"
python -m agentgym.cli rollout \
  --policy expert --incidents "$AGENTGYM_DATA/incidents_large.jsonl" \
  --split in_distribution \
  --out "$OUT/trajectories-$(printf '%03d' "$SLURM_ARRAY_TASK_ID").jsonl" \
  --gym-offset $((500 + SLURM_ARRAY_TASK_ID * 3)) --max-steps 28
echo "shard $SLURM_ARRAY_TASK_ID done"
