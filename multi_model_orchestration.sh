#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-80
#SBATCH --account=F202500001HPCVLABEPICUREG
#SBATCH --output=logs/deucalion/out/vllm_%j.out
#SBATCH --error=logs/deucalion/err/vllm_%j.err

set -euo pipefail

WORKDIR="/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench"
cd "$WORKDIR"

export HF_HOME="$WORKDIR/.cache/huggingface"
export SIF="med-llm-bench.sif"
mkdir -p "$HF_HOME"

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

declare -a RAY_PIDS=()

cleanup() {
    rc=$?
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
cat > configs/runtime/telemetry.auto.yaml <<EOF
telemetry:
  enabled: false
  collector: remote_http
  timeout_s: 1.5
  window_path: /window
  endpoints: []
EOF

export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export SINGULARITYENV_LLM_BENCH_RUNTIME_CONFIG="configs/runtime/telemetry.auto.yaml"
export SINGULARITYENV_LLM_MAX_TOKENS_DEFAULT="1024"
export SINGULARITYENV_SERVE_PORT="$SERVE_PORT"

mkdir -p logs/multi_model logs/vllm_serve outputs/reports outputs/raw

echo "Job $SLURM_JOB_ID on node $(hostname), nodes=${node_count}, gpus_per_node=${gpus_per_node}"

echo "Bootstrapping Ray cluster on allocated nodes..."
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

export SINGULARITYENV_VLLM_HOST_IP="$head_node_ip"
export SINGULARITYENV_LLM_BASE_URL="http://${head_node_ip}:${SERVE_PORT}/v1"

# Start Ray head
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --env-file .env $SIF \
    ray start --head --node-ip-address="$head_node_ip" --port="$RAY_PORT" --num-gpus="$gpus_per_node" --block &
RAY_PIDS+=("$!")
sleep 8

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

# Start telemetry agents on all nodes (head + workers)
telemetry_port=9101
for node in "${nodes_array[@]}"; do
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address | awk '{print $1}')
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        singularity exec --nv --env-file .env $SIF \
        python3 scripts/telemetry/nvidia_smi_agent.py --host 0.0.0.0 --port $telemetry_port --poll-interval-s 0.2 &
    sleep 2
    configs/runtime/telemetry.auto.yaml >/dev/null 2>&1 || true
done

python_script="scripts/multi_model_orchestrator.py"
echo "Running orchestrator: $python_script"

srun --overlap --nodes=1 --ntasks=1 \
    singularity exec --nv --env-file .env $SIF \
    python3 -u "$python_script"

EXIT_CODE=$?

echo "Orchestrator exited with code: $EXIT_CODE"

exit $EXIT_CODE
