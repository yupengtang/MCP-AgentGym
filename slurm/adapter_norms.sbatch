#!/bin/bash
#SBATCH --job-name=agentgym-norms
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Per-adapter ||dW||_F, for the question of whether update *magnitude* is what positions a
# model on the capability/generalization trade.
#
# CPU, not GPU: the computation is a 32x32 trace per pair and needs no accelerator. It is
# a batch job rather than a login-node command because the cost is loading ~160 MB of
# adapter per checkpoint off shared storage, which on a busy login node runs past any
# reasonable timeout -- two attempts were killed that way before this file existed.

set -euo pipefail
cd "${AGENTGYM_REPO:-$HOME/projects/Engineering/MCP-AgentGym}"
source env.sh

ADAPTERS="${ADAPTERS:-}"
if [ -z "$ADAPTERS" ]; then
  echo "set ADAPTERS to a space-separated list of adapter directories" >&2
  exit 2
fi
OUT="${OUT:-$AGENTGYM_RUNS/adapter-norms.json}"

echo "--- adapters ---"
for a in $ADAPTERS; do echo "  $a"; done

python scripts/adapter_norms.py $ADAPTERS --json-out "$OUT"
echo "norms -> $OUT"
