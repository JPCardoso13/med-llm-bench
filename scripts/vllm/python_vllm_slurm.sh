#!/bin/bash
# Submission wrapper for the rtx4060/haslab cluster. Run with `bash`, NOT
# `sbatch` - same pattern as slurm_orchestrator.sh: submits the shared job
# body (python_vllm_pipeline.sh) via sbatch, passing this cluster's resource
# requests as CLI flags. Usage:
#   bash scripts/vllm/python_vllm_slurm.sh <python_script> [args...]
set -euo pipefail

# rtx4060/haslab: internet access, small storage - export before sbatch so
# they reach the job (sbatch defaults to --export=ALL, propagating the
# submitting shell's environment). Still overridable by exporting these
# yourself before running this wrapper.
export WORKDIR="${WORKDIR:-/projects/jcardoso/med-llm-bench}"
export HF_OFFLINE="${HF_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs/python_vllm/out logs/python_vllm/err logs/vllm_setup

sbatch \
    --job-name=python_vllm \
    --partition=rtx4060 \
    --account=haslab \
    --nodes=2 \
    --time=02:00:00 \
    --output=logs/python_vllm/out/python_vllm_%j.out \
    --error=logs/python_vllm/err/python_vllm_%j.err \
    "$SCRIPT_DIR/python_vllm_pipeline.sh" "$@"
