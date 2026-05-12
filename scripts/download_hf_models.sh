#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=dev-x86
#SBATCH --account=F202500001HPCVLABEPICUREX
#SBATCH --output=logs/hf_download/out/hf_download_%j.out
#SBATCH --error=logs/hf_download/err/hf_download_%j.err

set -euo pipefail

WORKDIR="/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench"
cd "$WORKDIR"

SIF="med-llm-bench.sif"
HF_HOME="${HF_HOME:-$WORKDIR/.cache/huggingface}"

mkdir -p "$HF_HOME" logs/hf_download/out logs/hf_download/err

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HF_AUTH_TOKEN:-}}}"
export SINGULARITYENV_HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "Downloading models into: $HF_HOME"

singularity exec --env-file .env "$SIF" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-32B-AWQ",
]

# Use hub/ subdirectory to match huggingface_hub's offline cache structure
hf_home = os.environ["HF_HOME"]
cache_dir = os.path.join(hf_home, "hub")
os.makedirs(cache_dir, exist_ok=True)

for repo_id in models:
    local_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    print(f"{repo_id} -> {local_path}")
PY