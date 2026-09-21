# Source this before doing anything: `source env.sh`
#
# Site-specific paths belong in env.local.sh, which is untracked and sourced first if
# present. Everything below is a default that works on a fresh checkout; the only reason
# to override AGENTGYM_HOME is that runs, datasets and model caches are large and a
# cluster home directory usually has a quota.
#
#   cp env.local.sh.example env.local.sh   # then edit

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_here/env.local.sh" ] && . "$_here/env.local.sh"

export AGENTGYM_REPO="${AGENTGYM_REPO:-$_here}"
export AGENTGYM_HOME="${AGENTGYM_HOME:-$HOME/.agentgym}"

# Kubernetes control-plane binaries (etcd, kube-apiserver, ...)
export AGENTGYM_BIN="${AGENTGYM_BIN:-$AGENTGYM_HOME/bin}"

# Episode state lives on node-local disk: NFS is far too slow for etcd.
export AGENTGYM_RUNTIME="${AGENTGYM_RUNTIME:-${TMPDIR:-/tmp}/$USER/agentgym-runtime}"

# Durable artifacts
export AGENTGYM_RUNS="${AGENTGYM_RUNS:-$AGENTGYM_HOME/runs}"
export AGENTGYM_DATA="${AGENTGYM_DATA:-$AGENTGYM_HOME/data}"
export AGENTGYM_ARTIFACTS="${AGENTGYM_ARTIFACTS:-$AGENTGYM_HOME/artifacts}"

export HF_HOME="${HF_HOME:-$AGENTGYM_HOME/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Two environments, deliberately separate. vLLM pins transformers v5 and a CUDA 13 torch;
# peft still requires transformers v4, and a 575.x / CUDA 12.9 driver cannot run CUDA 13
# wheels. Training and serving therefore get their own venvs, both built against cu128.
export AGENTGYM_VENV="${AGENTGYM_VENV:-$AGENTGYM_HOME/venv}"            # torch 2.8+cu128, transformers 4, trl, peft
export AGENTGYM_VLLM_VENV="${AGENTGYM_VLLM_VENV:-$AGENTGYM_HOME/venv-vllm}"  # vllm 0.11.0, torch 2.8+cu128

export MLFLOW_TRACKING_URI="file://$AGENTGYM_HOME/mlruns"
# MLflow deprecated the file store, but it is the right backend here: runs are
# directory-per-run, so concurrent Slurm jobs on shared NFS never contend for a lock the
# way they would on SQLite.
export MLFLOW_ALLOW_FILE_STORE=true
export UV_CACHE_DIR="${UV_CACHE_DIR:-$AGENTGYM_HOME/cache/uv}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$AGENTGYM_REPO/src:${PYTHONPATH:-}"
export PATH="$AGENTGYM_BIN:$AGENTGYM_VENV/bin:$PATH"

mkdir -p "$AGENTGYM_RUNTIME" "$AGENTGYM_RUNS" "$AGENTGYM_DATA" "$AGENTGYM_ARTIFACTS"
unset _here

echo "agentgym env: HOME=$AGENTGYM_HOME RUNTIME=$AGENTGYM_RUNTIME python=$(command -v python)"
