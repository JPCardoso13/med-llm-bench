#!/bin/bash
# Shared SLURM job body for launching an ad-hoc Python script inside the
# vLLM/Ray container environment (e.g. one-off vLLM smoke tests, GPU-side
# preprocessing). Submitted via sbatch by a per-cluster wrapper
# (python_vllm_deucalion.sh for A100/deucalion, python_vllm_slurm.sh for
# rtx4060/haslab) - those wrappers pass all cluster-specific resource
# requests (partition, account, node count, time limit, output paths) as
# sbatch CLI flags rather than #SBATCH pragmas here, mirroring
# run_pipeline.sh/deucalion_orchestrator.sh/slurm_orchestrator.sh. Do not
# submit this file directly with sbatch; run a wrapper with bash instead.
# WORKDIR/HF_OFFLINE are cluster-specific and are set by each wrapper before
# it calls sbatch, not here.

set -euo pipefail

PYTHON_SCRIPT="$1"
shift
PYTHON_ARGS=("$@")

WORKDIR="${WORKDIR:?WORKDIR is not set - run this via python_vllm_deucalion.sh or python_vllm_slurm.sh, which set it per-cluster (do not submit python_vllm_pipeline.sh directly)}"
cd "$WORKDIR"

SIF="med-llm-bench.sif"
# The populated HF cache lives outside the project directory (a sibling of
# WORKDIR, not inside it) - see run_pipeline.sh for the same fix.
export HF_HOME="${HF_HOME:-$(dirname "$WORKDIR")/.cache/huggingface}"
mkdir -p "$HF_HOME" logs/vllm_setup

HF_OFFLINE="${HF_OFFLINE:-0}"
if [[ "$HF_OFFLINE" == "1" ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
else
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
fi

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
export SINGULARITYENV_HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"
export SINGULARITYENV_PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"

RAY_PORT="${RAY_PORT:-6379}"
node_count="${SLURM_NNODES:-1}"

if [[ "$node_count" -gt 1 ]]; then
    mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    head_node="${nodes_array[0]}"
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
    gpus_per_node=$(echo "${SLURM_GPUS_PER_NODE:-1}" | grep -oE '[0-9]+' | head -n1)

    echo "Starting Ray head on $head_node ($head_node_ip)"
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        singularity exec --nv --env-file .env "$SIF" \
        ray start --head --node-ip-address="$head_node_ip" --port="$RAY_PORT" --num-gpus="$gpus_per_node" --block &
    sleep 8

    for ((i = 1; i < ${#nodes_array[@]}; i++)); do
        worker_node="${nodes_array[$i]}"
        worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address | awk '{print $1}')
        echo "Starting Ray worker on $worker_node ($worker_ip)"
        srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
            singularity exec --nv --env-file .env "$SIF" \
            ray start --address="${head_node_ip}:${RAY_PORT}" --node-ip-address="$worker_ip" --num-gpus="$gpus_per_node" --block &
        sleep 5
    done

    export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"
    export SINGULARITYENV_RAY_ADDRESS="$RAY_ADDRESS"

    echo "Ray cluster ready at $RAY_ADDRESS"
fi

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname) | nodes=${node_count} | offline=${HF_OFFLINE}"
echo "Running: $PYTHON_SCRIPT ${PYTHON_ARGS[*]:-}"

singularity exec --nv --env-file .env "$SIF" \
    python3 -u "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
