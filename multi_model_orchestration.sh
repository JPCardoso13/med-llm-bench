#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=F202500001HPCVLABEPICUREG
#SBATCH --output=logs/orchestration/out/multi_model_orchestration_%j.out
#SBATCH --error=logs/orchestration/err/multi_model_orchestration_%j.err

# Suggested defaults:
# - No internet, big storage: HF_OFFLINE=1 HF_CACHE_MODE=persistent HF_EVICT_BETWEEN_MODELS=0
# - Internet, small storage: HF_OFFLINE=0 HF_CACHE_MODE=ephemeral HF_EVICT_BETWEEN_MODELS=1

set -euo pipefail

WORKDIR="/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench"
cd "$WORKDIR"

export SIF="med-llm-bench.sif"

HF_CACHE_MODE="${HF_CACHE_MODE:-persistent}"
if [[ "$HF_CACHE_MODE" == "ephemeral" ]]; then
    job_tmp_root="${SLURM_TMPDIR:-$WORKDIR/tmp/slurm_${SLURM_JOB_ID:-manual}}"
    export HF_HOME="$job_tmp_root/huggingface"
else
    export HF_HOME="${HF_HOME:-$WORKDIR/.cache/huggingface}"
fi
mkdir -p "$HF_HOME"

HF_OFFLINE="${HF_OFFLINE:-0}"
if [[ "$HF_OFFLINE" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    export HF_DATASETS_OFFLINE=0
fi

HF_EVICT_BETWEEN_MODELS="${HF_EVICT_BETWEEN_MODELS:-0}"

SERVE_PORT="${SERVE_PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"

extract_first_int() {
    local value="$1"
    local out
    out=$(echo "$value" | grep -oE '[0-9]+' | head -n1 || true)
    if [[ -z "$out" ]]; then
        echo "1"
    else
        echo "$out"
    fi
}

node_count=$(extract_first_int "${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}")
gpus_per_node=$(extract_first_int "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}")
multi_node=0
if [[ "$node_count" -gt 1 ]]; then
    multi_node=1
fi

declare -a RAY_PIDS=()
declare -a TELEMETRY_PIDS=()

cleanup() {
    rc=$?
    if [[ "$HF_CACHE_MODE" == "ephemeral" ]]; then
        echo "Cleaning ephemeral Hugging Face cache at $HF_HOME"
        rm -rf "$HF_HOME" >/dev/null 2>&1 || true
    fi
    if [[ ${#TELEMETRY_PIDS[@]} -gt 0 ]]; then
        echo "Stopping telemetry processes..."
        for pid in "${TELEMETRY_PIDS[@]}"; do
            kill "$pid" >/dev/null 2>&1 || true
        done
    fi
    if [[ ${#RAY_PIDS[@]} -gt 0 ]]; then
        echo "Stopping Ray processes..."
        for pid in "${RAY_PIDS[@]}"; do
            kill "$pid" >/dev/null 2>&1 || true
        done
    fi
    exit $rc
}
trap cleanup EXIT

mkdir -p configs/runtime

export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
export SINGULARITYENV_HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
export SINGULARITYENV_HF_OFFLINE="$HF_OFFLINE"
export SINGULARITYENV_HF_EVICT_BETWEEN_MODELS="$HF_EVICT_BETWEEN_MODELS"
export SINGULARITYENV_PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export SINGULARITYENV_LLM_BENCH_RUNTIME_CONFIG="configs/runtime/telemetry.auto.yaml"
export SINGULARITYENV_LLM_MAX_TOKENS_DEFAULT="1024"
export SINGULARITYENV_LLM_NODE_COUNT="$node_count"
export SINGULARITYENV_SERVE_PORT="$SERVE_PORT"

mkdir -p logs/orchestration/out logs/orchestration/err logs/vllm outputs/reports outputs/raw

echo "Job $SLURM_JOB_ID on node $(hostname), nodes=${node_count}, gpus_per_node=${gpus_per_node}"

if [[ "$multi_node" -eq 1 ]]; then
    echo "Bootstrapping Ray cluster on allocated nodes..."
    mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    head_node="${nodes_array[0]}"
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

    export SINGULARITYENV_VLLM_HOST_IP="$head_node_ip"
    export SINGULARITYENV_LLM_BASE_URL="http://${head_node_ip}:${SERVE_PORT}/v1"

    # Start Ray head
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP="$head_node_ip" \
        singularity exec --nv --env-file .env $SIF \
        ray start --head --node-ip-address="$head_node_ip" --port="$RAY_PORT" --num-gpus="$gpus_per_node" --block &
    RAY_PIDS+=("$!")
    sleep 8

    for ((i=1; i<${#nodes_array[@]}; i++)); do
        worker_node="${nodes_array[$i]}"
        worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address | awk '{print $1}')

        srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
            env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP="$worker_ip" \
            singularity exec --nv --env-file .env $SIF \
            ray start --address="${head_node_ip}:${RAY_PORT}" --node-ip-address="$worker_ip" --num-gpus="$gpus_per_node" --block &
        RAY_PIDS+=("$!")
        sleep 5
    done

    export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"
    export SINGULARITYENV_RAY_ADDRESS="$RAY_ADDRESS"
else
    echo "Single-node allocation detected; skipping Ray bootstrap."
fi

# Start telemetry agents on all nodes (head + workers)
telemetry_port=9101
declare -a TELEMETRY_ENDPOINTS=()
for node in "${nodes_array[@]}"; do
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address | awk '{print $1}')
    TELEMETRY_ENDPOINTS+=("http://${node_ip}:${telemetry_port}")
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        singularity exec --nv --env-file .env $SIF \
        python3 scripts/telemetry/nvidia_smi_agent.py --host 0.0.0.0 --port $telemetry_port --poll-interval-s 0.2 &
    TELEMETRY_PIDS+=("$!")
    sleep 2
done

{
    echo "telemetry:"
    echo "  enabled: true"
    echo "  collector: remote_http"
    echo "  timeout_s: 1.5"
    echo "  window_path: /window"
    echo "  endpoints:"
    for endpoint in "${TELEMETRY_ENDPOINTS[@]}"; do
        echo "    - ${endpoint}"
    done
} > configs/runtime/telemetry.auto.yaml

echo "Telemetry endpoints configured: ${#TELEMETRY_ENDPOINTS[@]} node(s)"
echo "Cache mode: ${HF_CACHE_MODE} | HF_HOME=${HF_HOME} | Offline=${HF_OFFLINE}"

python_script="scripts/multi_model_orchestrator.py"
echo "Running orchestrator: $python_script"

srun --overlap --nodes=1 --ntasks=1 \
    singularity exec --nv --env-file .env $SIF \
    python3 -u "$python_script"

EXIT_CODE=$?

echo "Orchestrator exited with code: $EXIT_CODE"

exit $EXIT_CODE
