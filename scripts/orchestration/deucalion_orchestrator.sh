#!/bin/bash
# Submission wrapper for the A100/deucalion cluster. Run with `bash`, NOT
# `sbatch` - see slurm_orchestrator.sh for why (same shared job body,
# run_pipeline.sh, just different cluster-specific resource flags below).
set -euo pipefail

# A100/deucalion: no internet access, big persistent storage - export before
# sbatch so they reach the job (sbatch defaults to --export=ALL, propagating
# the submitting shell's environment). Still overridable by exporting these
# yourself before running this wrapper.
export WORKDIR="${WORKDIR:-/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench}"
export HF_OFFLINE="${HF_OFFLINE:-1}"
export HF_CACHE_MODE="${HF_CACHE_MODE:-persistent}"
export HF_EVICT_BETWEEN_MODELS="${HF_EVICT_BETWEEN_MODELS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch \
    --job-name=deucalion_orchestrator \
    --partition=normal-a100-80 \
    --account=F202500001HPCVLABEPICUREG \
    --nodes=1 \
    --gpus=4 \
    --ntasks=1 \
    --cpus-per-task=128 \
    --time=06:00:00 \
    --output=logs/orchestration/out/deucalion_orchestrator_%j.out \
    --error=logs/orchestration/err/deucalion_orchestrator_%j.err \
    "$SCRIPT_DIR/run_pipeline.sh"
