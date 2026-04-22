#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH -p rtx4060
#SBATCH -A haslab
#SBATCH --nodes=2
#SBATCH --exclude=aurora[04-05]
#SBATCH --time=02:00:00
#SBATCH --output=logs/vllm_serve/vllm_%j.log

export HF_HOME=/projects/jcardoso/med-llm-bench/.cache/huggingface
export SIF="med-llm-bench.sif"

MODEL_CONFIG="${MODEL_CONFIG:-configs/models/llama3_8b_instruct.yaml}"

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Model config not found: $MODEL_CONFIG" >&2
    exit 1
fi

yaml_get() {
    local key="$1"
    local file="$2"
    awk -F': ' -v k="$key" '$1 == k {print $2; exit}' "$file" | sed 's/^"//; s/"$//'
}

MODEL_ID="$(yaml_get model_id "$MODEL_CONFIG")"
TP_SIZE="$(yaml_get tensor_parallel_size "$MODEL_CONFIG")"
MAX_MODEL_LEN="$(yaml_get max_model_len "$MODEL_CONFIG")"
GPU_MEMORY_UTILIZATION="$(yaml_get gpu_memory_utilization "$MODEL_CONFIG")"
ENFORCE_EAGER="$(yaml_get enforce_eager "$MODEL_CONFIG")"

if [[ -z "$MODEL_ID" ]]; then
    echo "model_id is required in $MODEL_CONFIG" >&2
    exit 1
fi

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
node_count=${#nodes_array[@]}

if [[ "$node_count" -lt 1 ]]; then
    echo "No nodes allocated by Slurm." >&2
    exit 1
fi

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port=6379
serve_port=8000
telemetry_port=9101
telemetry_poll_interval=0.2
telemetry_gpu_indices="${TELEMETRY_GPU_INDICES:-}"

mkdir -p configs/runtime

echo "Starting Ray Head on $head_node ($head_node_ip)"

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP=$head_node_ip \
    singularity exec --nv $SIF \
    ray start --head --node-ip-address=$head_node_ip --port=$port --num-gpus=1 --block &
sleep 10

declare -a node_ips
node_ips=("$head_node_ip")

for ((i=1; i<node_count; i++)); do
    worker_node=${nodes_array[$i]}
    worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address | awk '{print $1}')
    node_ips+=("$worker_ip")

    echo "Starting Ray Worker on $worker_node ($worker_ip)"
    srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
        env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP=$worker_ip \
        singularity exec --nv $SIF \
        ray start --address="$head_node_ip:$port" --node-ip-address=$worker_ip --num-gpus=1 --block &
    sleep 10
done

for ((i=0; i<node_count; i++)); do
    node=${nodes_array[$i]}
    node_ip=${node_ips[$i]}

    telemetry_gpu_args=()
    if [[ -n "$telemetry_gpu_indices" ]]; then
        telemetry_gpu_args+=(--gpu-indices "$telemetry_gpu_indices")
    fi

    echo "Starting telemetry agent on $node ($node_ip:$telemetry_port)"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        singularity exec --nv $SIF \
        python3 scripts/telemetry/nvidia_smi_agent.py \
            --host 0.0.0.0 \
            --port $telemetry_port \
            --poll-interval-s $telemetry_poll_interval \
            "${telemetry_gpu_args[@]}" &
    sleep 3
done

cat > configs/runtime/telemetry.auto.yaml <<EOF
telemetry:
  enabled: true
  collector: remote_http
  timeout_s: 1.5
  window_path: /window
  endpoints:
EOF

for ip in "${node_ips[@]}"; do
        echo "    - http://${ip}:${telemetry_port}" >> configs/runtime/telemetry.auto.yaml
done

echo "Wrote runtime telemetry config to configs/runtime/telemetry.auto.yaml"

export RAY_ADDRESS="${head_node_ip}:${port}"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_VLLM_HOST_IP=$head_node_ip
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "Starting vLLM server on ${head_node_ip}:${serve_port}..."

enforce_eager_args=()
if [[ "${ENFORCE_EAGER:-true}" == "true" ]]; then
    enforce_eager_args+=("--enforce-eager")
fi

singularity exec --nv --env-file .env $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --tensor-parallel-size "${TP_SIZE:-$node_count}" \
        --distributed-executor-backend ray \
        --host 0.0.0.0 \
        --port $serve_port \
        --max-model-len "${MAX_MODEL_LEN:-4096}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
        "${enforce_eager_args[@]}"