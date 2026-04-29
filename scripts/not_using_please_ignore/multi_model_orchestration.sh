#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition normal-a100-40
#SBATCH --account=F202500001HPCVLABEPICUREG
#SBATCH --output=logs/multi_model/multi_model_%j.log

# Multi-model benchmarking orchestrator (deferred metrics only)
#
# Resource sizing is controlled via #SBATCH headers above.
# This script:
# 1. Discovers model configs in configs/models/*.yaml
# 2. For each model, starts a serving process with single-node first
# 3. Falls back to distributed serving (Ray) when available and needed
# 4. Captures raw measurements for all models
# 5. Computes metrics only after all model runs complete

set -e

WORKDIR="/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench"
cd "$WORKDIR"

export HF_HOME=/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench/.cache/huggingface
export SIF="med-llm-bench.sif"
mkdir -p "$HF_HOME"

SERVE_PORT="${SERVE_PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"

extract_first_int() {
    local value="$1"
    local out
    out=$(echo "$value" | grep -oE '[0-9]+' | head -n1)
    if [[ -z "$out" ]]; then
        echo "1"
    else
        echo "$out"
    fi
}

host_ip=$(hostname -I | awk '{print $1}')
node_count=$(extract_first_int "${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}")
gpus_per_node=$(extract_first_int "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}")

declare -a RAY_PIDS=()

cleanup() {
    local exit_code=$?
    if [[ ${#RAY_PIDS[@]} -gt 0 ]]; then
        echo "Stopping Ray processes..."
        for pid in "${RAY_PIDS[@]}"; do
            kill "$pid" >/dev/null 2>&1 || true
        done
    fi
    exit "$exit_code"
}

trap cleanup EXIT

mkdir -p configs/runtime
cat > configs/runtime/telemetry.auto.yaml <<EOF
telemetry:
  enabled: false
EOF

export SINGULARITYENV_VLLM_HOST_IP=$host_ip
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_LLM_BASE_URL="http://${host_ip}:${SERVE_PORT}/v1"
export SINGULARITYENV_LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export SINGULARITYENV_LLM_BENCH_RUNTIME_CONFIG="configs/runtime/telemetry.auto.yaml"
export SINGULARITYENV_LLM_MAX_TOKENS_DEFAULT="1024"
export SINGULARITYENV_SERVE_PORT="$SERVE_PORT"

mkdir -p logs/multi_model logs/vllm_serve outputs/reports

echo "Job $SLURM_JOB_ID on node $(hostname), nodes=${node_count}, gpus_per_node=${gpus_per_node}"
echo "vLLM endpoint: ${SINGULARITYENV_LLM_BASE_URL}"

echo "Bootstrapping Ray cluster on allocated nodes..."
    echo "Multi-node allocation detected. Bootstrapping Ray cluster..."

    mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    head_node="${nodes_array[0]}"
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        singularity exec --nv --env-file .env $SIF \
        ray start --head --node-ip-address="$head_node_ip" --port="$RAY_PORT" --num-gpus="$gpus_per_node" --block &
    RAY_PIDS+=("$!")
    sleep 10

    for ((i=1; i<${#nodes_array[@]}; i++)); do
        worker_node="${nodes_array[$i]}"
        worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address | awk '{print $1}')

        srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
            singularity exec --nv --env-file .env $SIF \
            ray start --address="${head_node_ip}:${RAY_PORT}" --node-ip-address="$worker_ip" --num-gpus="$gpus_per_node" --block &
        RAY_PIDS+=("$!")
        sleep 5
    done

    export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"
    export SINGULARITYENV_RAY_ADDRESS="$RAY_ADDRESS"
    export SINGULARITYENV_VLLM_HOST_IP="$head_node_ip"
    export SINGULARITYENV_LLM_BASE_URL="http://${head_node_ip}:${SERVE_PORT}/v1"
    echo "Ray cluster ready at ${RAY_ADDRESS}"

ORCHESTRATOR_SCRIPT="scripts/multi_model_orchestrator.py"

echo "Running multi-model orchestration (per-model eager metrics) against ${SINGULARITYENV_LLM_BASE_URL}"
echo "Script: ${ORCHESTRATOR_SCRIPT}"

srun --overlap --nodes=1 --ntasks=1 \
    singularity exec --nv --env-file .env $SIF \
    python3 -u "$ORCHESTRATOR_SCRIPT"

EXIT_CODE=$?

echo "Multi-model orchestration completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
