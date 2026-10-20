#!/bin/bash
#SBATCH --job-name=agentgym-expert
#SBATCH --partition=cpu
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out

# Expert trajectory generation. CPU only: the environment is a real Kubernetes control
# plane plus a deterministic simulator, and the expert is a scripted policy, so no GPU is
# involved. Each array task owns one control plane and its own port block.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh

# Node-local disk for etcd: NFS latency would dominate episode time.
export AGENTGYM_RUNTIME="${TMPDIR:-/tmp}/$USER/agentgym-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$AGENTGYM_RUNTIME"
trap 'rm -rf "$AGENTGYM_RUNTIME"' EXIT

PERTURBATION="${PERTURBATION:-clean}"
OUTDIR="${OUTDIR:-$AGENTGYM_RUNS/expert-$PERTURBATION}"
mkdir -p "$OUTDIR"

# Optional alternate incident pool. A held-out set other than `published` produces a
# different corpus, and demonstrations for it have to come from that pool -- rolling out
# the default pool would put withheld families back into the training data, which is
# precisely the thing the split exists to prevent and would not show up anywhere except
# as a held-out score that was implausibly good.
INCIDENTS="${INCIDENTS:-}"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname) runtime=$AGENTGYM_RUNTIME"
echo "  pool=${INCIDENTS:-<default>}  heldout_set=${AGENTGYM_HELDOUT_SET:-published}"

python -m agentgym.cli rollout \
  --policy expert \
  --perturbation "$PERTURBATION" \
  ${INCIDENTS:+--incidents "$INCIDENTS"} \
  --out "$OUTDIR/trajectories-$(printf '%03d' "$SLURM_ARRAY_TASK_ID").jsonl" \
  --gym-offset $((200 + SLURM_ARRAY_TASK_ID * 3)) \
  --max-steps 28

echo "task $SLURM_ARRAY_TASK_ID done"
