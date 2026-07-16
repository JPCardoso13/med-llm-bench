#!/bin/bash
# Submission wrapper for the rtx4060/haslab cluster. Run with `bash`, NOT
# `sbatch` - this script submits the real job (run_pipeline.sh, shared with
# deucalion_orchestrator.sh) via sbatch, passing this cluster's resource
# requests as CLI flags (sbatch flags override any #SBATCH pragmas in the
# target script, so the shared job body carries none - keeping cluster-
# specific values in exactly one place per cluster instead of duplicated
# across two ~220-line near-identical job scripts).
set -euo pipefail

# rtx4060/haslab: internet access, small storage - export before sbatch so
# they reach the job (sbatch defaults to --export=ALL, propagating the
# submitting shell's environment). Still overridable by exporting these
# yourself before running this wrapper.
export HF_OFFLINE="${HF_OFFLINE:-0}"
export HF_CACHE_MODE="${HF_CACHE_MODE:-ephemeral}"
export HF_EVICT_BETWEEN_MODELS="${HF_EVICT_BETWEEN_MODELS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch \
    --job-name=slurm_orchestrator \
    --partition=rtx4060 \
    --account=haslab \
    --nodes=2 \
    --exclude=aurora[04-05] \
    --time=02:00:00 \
    --output=logs/orchestration/out/slurm_orchestrator_%j.out \
    --error=logs/orchestration/err/slurm_orchestrator_%j.err \
    "$SCRIPT_DIR/run_pipeline.sh"
