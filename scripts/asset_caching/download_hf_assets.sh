#!/bin/bash
#SBATCH --job-name=download_hf_assets
#SBATCH --partition=normal-a100-80
#SBATCH --account=F202500001HPCVLABEPICUREG
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=logs/asset_download/out/download_hf_assets_%j.out
#SBATCH --error=logs/asset_download/err/download_hf_assets_%j.err

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