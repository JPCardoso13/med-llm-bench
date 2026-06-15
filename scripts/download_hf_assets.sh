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

mkdir -p "$HF_HOME" "$HF_HOME/hub" "$HF_HOME/datasets" logs/hf_download/out logs/hf_download/err

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_HF_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_HOME/datasets"
export SINGULARITYENV_HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HF_AUTH_TOKEN:-}}}"
export SINGULARITYENV_HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "Downloading Hugging Face assets into: $HF_HOME"

singularity exec --env-file .env "$SIF" python3 scripts/download_hf_assets.py