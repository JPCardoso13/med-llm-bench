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
export WORKDIR="${WORKDIR:-/projects/jcardoso/med-llm-bench}"
export HF_OFFLINE="${HF_OFFLINE:-0}"
export HF_CACHE_MODE="${HF_CACHE_MODE:-ephemeral}"
export HF_EVICT_BETWEEN_MODELS="${HF_EVICT_BETWEEN_MODELS:-1}"

# run_pipeline.sh puts the ephemeral HF cache under ${SLURM_TMPDIR:-$WORKDIR/tmp/...}
# - this cluster's SLURM does not set SLURM_TMPDIR itself (confirmed empty in
# a real job), so without this the "ephemeral" cache was actually landing on
# the 50GB-quota /projects filesystem the whole time. Each node has its own
# ~148GB local /tmp (confirmed via srun on aurora03/aurora09) that isn't
# quota-tracked - point SLURM_TMPDIR there instead. Since /tmp is node-local
# (not shared across the allocation), every node independently downloads its
# own copy of whatever it needs under this same path string, which is exactly
# what multi-node tensor-parallel serving requires anyway.
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/med-llm-bench-hf-$$}"

# Defaults match the 2-node/3h local_smoke validation scale. Override before
# invoking this wrapper (e.g. NODES=8 TIME_LIMIT=16:00:00 bash ...) for runs
# whose model roster needs a larger tensor_parallel_size or more wall time -
# the definitive rtx4060_production.yaml roster needs NODES=8 (its two 27B
# BF16 models require tensor_parallel_size: 8).
NODES="${NODES:-2}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch \
    --job-name=slurm_orchestrator \
    --partition=rtx4060 \
    --account=haslab \
    --nodes="$NODES" \
    --time="$TIME_LIMIT" \
    --output=logs/orchestration/out/slurm_orchestrator_%j.out \
    --error=logs/orchestration/err/slurm_orchestrator_%j.err \
    "$SCRIPT_DIR/run_pipeline.sh"
