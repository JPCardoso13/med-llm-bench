#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition normal-a100-40
#SBATCH --account=F202500001HPCVLABEPICUREG
#SBATCH --output=logs/deucalion/out/vllm_%j.out
#SBATCH --error=logs/deucalion/err/vllm_%j.err

set -e

WORKDIR="/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench"
cd "$WORKDIR"

export HF_HOME=/projects/F202500001HPCVLABEPICURE/jcardoso/med-llm-bench/.cache/huggingface
export SIF="med-llm-bench.sif"
mkdir -p "$HF_HOME"

MODEL_CONFIG="${MODEL_CONFIG:-configs/models/qwen3_32b_awq.yaml}"
GPU_INDEX="${GPU_INDEX:-0}"
SERVE_PORT="${SERVE_PORT:-8000}"

yaml_get() {
    local key="$1"
    local file="$2"
    awk -F': ' -v k="$key" '$1 == k {print $2; exit}' "$file" | sed 's/^"//; s/"$//'
}

MODEL_ID="$(yaml_get model_id "$MODEL_CONFIG")"
MAX_MODEL_LEN="$(yaml_get max_model_len "$MODEL_CONFIG")"
GPU_MEMORY_UTILIZATION="$(yaml_get gpu_memory_utilization "$MODEL_CONFIG")"
ENFORCE_EAGER="$(yaml_get enforce_eager "$MODEL_CONFIG")"

TP_SIZE=1

host_ip=$(hostname -I | awk '{print $1}')

mkdir -p configs/runtime
cat > configs/runtime/telemetry.auto.yaml <<EOF
telemetry:
  enabled: false
EOF

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_INDEX
export SINGULARITYENV_VLLM_HOST_IP=$host_ip
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export SINGULARITYENV_PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_LLM_BASE_URL="http://${host_ip}:${SERVE_PORT}/v1"
export SINGULARITYENV_LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export SINGULARITYENV_LLM_BENCH_RUNTIME_CONFIG="configs/runtime/telemetry.auto.yaml"
export SINGULARITYENV_LLM_BENCH_MODEL_CONFIG="$MODEL_CONFIG"
export SINGULARITYENV_LLM_MAX_TOKENS_DEFAULT="1024"

mkdir -p logs/vllm_serve
server_log="logs/vllm_serve/vllm_${SLURM_JOB_ID}.log"

echo "Job $SLURM_JOB_ID on node $(hostname), GPU index ${GPU_INDEX}"
echo "Serving model $MODEL_ID at ${SINGULARITYENV_LLM_BASE_URL}"

enforce_eager_args=()
if [[ "${ENFORCE_EAGER:-true}" == "true" ]]; then
    enforce_eager_args+=("--enforce-eager")
fi

srun --overlap --nodes=1 --ntasks=1 \
    env SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_INDEX SINGULARITYENV_VLLM_HOST_IP=$host_ip \
    singularity exec --nv --env-file .env $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --tensor-parallel-size "$TP_SIZE" \
        --host 0.0.0.0 \
        --port $SERVE_PORT \
        --max-model-len "${MAX_MODEL_LEN:-4096}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
        "${enforce_eager_args[@]}" >"$server_log" 2>&1 &
server_pid=$!

probe_vllm_endpoint() {
    PROBE_URL="${SINGULARITYENV_LLM_BASE_URL}/models" \
    singularity exec --nv --env-file .env $SIF \
        python3 - <<'PY'
import os
import sys
import urllib.request

url = os.environ["PROBE_URL"]
try:
    response = urllib.request.urlopen(url, timeout=2)
    sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

echo "Waiting for vLLM endpoint..."
while true; do
    if probe_vllm_endpoint; then
        echo "vLLM endpoint is ready"
        break
    fi

    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "vLLM process exited before becoming ready. Check $server_log" >&2
        exit 1
    fi

    sleep 2
done

echo "Running CDKR benchmark against ${SINGULARITYENV_LLM_BASE_URL}"
srun --overlap --nodes=1 --ntasks=1 \
    singularity exec --nv --env-file .env $SIF \
    python3 -u scripts/cdkr_test_run.py

kill "$server_pid" >/dev/null 2>&1 || true