# Source this before doing anything: `source env.sh`
# Keeps all heavy artifacts off the 30GB home quota.

export AGENTGYM_HOME="${AGENTGYM_HOME:-/storage/ice-shared/mgt6655/storage/mcp-agentgym}"
export AGENTGYM_REPO="${AGENTGYM_REPO:-/home/hice1/ytang454/projects/Engineering/MCP-AgentGym}"

# Kubernetes control-plane binaries (etcd, kube-apiserver, ...)
export AGENTGYM_BIN="$AGENTGYM_HOME/bin"

# Episode state lives on node-local disk: NFS is far too slow for etcd.
export AGENTGYM_RUNTIME="${AGENTGYM_RUNTIME:-${TMPDIR:-/tmp}/$USER/agentgym-runtime}"

# Durable artifacts
export AGENTGYM_RUNS="$AGENTGYM_HOME/runs"
export AGENTGYM_DATA="$AGENTGYM_HOME/data"
export AGENTGYM_ARTIFACTS="$AGENTGYM_HOME/artifacts"

# Model cache (Qwen3-8B base + Qwen3-32B teacher already present)
export HF_HOME="${HF_HOME:-/storage/ice-shared/mgt6655/storage/ytang454_act_council/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"

# Two environments, deliberately separate. vLLM pins transformers v5 and a CUDA 13
# torch; peft still requires transformers v4, and the PACE driver (575.x / CUDA 12.9)
# cannot run CUDA 13 wheels. Training and serving therefore get their own venvs, both
# built against cu128.
export AGENTGYM_VENV="$AGENTGYM_HOME/venv"            # torch 2.8+cu128, transformers 4, trl, peft
export AGENTGYM_VLLM_VENV="$AGENTGYM_HOME/venv-vllm"  # vllm 0.11.0, torch 2.8+cu128

export MLFLOW_TRACKING_URI="file://$AGENTGYM_HOME/mlruns"
# MLflow deprecated the file store, but it is the right backend here: runs are
# directory-per-run, so concurrent Slurm jobs on shared NFS never contend for a lock the
# way they would on SQLite.
export MLFLOW_ALLOW_FILE_STORE=true
export UV_CACHE_DIR="$AGENTGYM_HOME/cache/uv"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$AGENTGYM_REPO/src:${PYTHONPATH:-}"
export PATH="$AGENTGYM_BIN:$AGENTGYM_HOME/venv/bin:$PATH"

mkdir -p "$AGENTGYM_RUNTIME" "$AGENTGYM_RUNS" "$AGENTGYM_DATA" "$AGENTGYM_ARTIFACTS"

echo "agentgym env: HOME=$AGENTGYM_HOME RUNTIME=$AGENTGYM_RUNTIME python=$(command -v python)"
