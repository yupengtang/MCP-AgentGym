#!/usr/bin/env bash
# Restore the HuggingFace model weights deleted on 2026-07-25 from
#   /storage/ice-shared/mgt6655/storage/ytang454_act_council/hf_cache
#
# All of these are public HF repos, so they can be re-downloaded in full.
# NOTE: needs ~175 GB free. Check `df -h /storage/ice-shared/mgt6655` first.
#
# Usage:
#   bash restore_deleted_models.sh            # restore everything
#   bash restore_deleted_models.sh Qwen/Qwen3-14B mistralai/Mistral-Small-3.2-24B-Instruct-2506
set -euo pipefail

export HF_HOME=/storage/ice-shared/mgt6655/storage/ytang454_act_council/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"

ALL_MODELS=(
  "mistralai/Mistral-Small-3.2-24B-Instruct-2506"   # ~90G
  "Qwen/Qwen3-14B"                                  # ~28G
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"        # ~28G
  "meta-llama/Meta-Llama-3.1-8B-Instruct"           # ~15G (gated: needs `hf auth login`)
  "Qwen/Qwen2.5-72B-Instruct"                       # ~14G (was a PARTIAL download)
)

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then MODELS=("${ALL_MODELS[@]}"); fi

if command -v hf >/dev/null 2>&1;      then DL="hf download";
elif command -v huggingface-cli >/dev/null 2>&1; then DL="huggingface-cli download";
else echo "ERROR: install the HF CLI first:  pip install -U 'huggingface_hub[cli]'" >&2; exit 1; fi

echo "HF_HUB_CACHE=$HF_HUB_CACHE"
for m in "${MODELS[@]}"; do
  echo "=== restoring $m ==="
  $DL "$m" || echo "!! FAILED: $m (gated repo? run 'hf auth login')" >&2
done
echo "done. current cache:"
du -sh "$HF_HUB_CACHE"/models--* 2>/dev/null || true
