#!/bin/bash
# Shared SLURM job body for the full benchmarking pipeline (vLLM serving +
# orchestrator + judge pass), submitted via sbatch by a per-cluster wrapper
# (slurm_orchestrator.sh for rtx4060/haslab, deucalion_orchestrator.sh for
# A100/deucalion) - those wrappers pass all cluster-specific resource
# requests (partition, account, node/gpu counts, time limit, output paths)
# as sbatch CLI flags rather than #SBATCH pragmas here, since CLI flags
# override pragmas. Do not submit this file directly with sbatch; run a
# wrapper with bash instead. HF_OFFLINE/HF_CACHE_MODE/HF_EVICT_BETWEEN_MODELS
# are cluster-specific (network access, storage size) and are set by each
# wrapper before it calls sbatch, not here - the fallbacks below only apply
# if this is somehow invoked without going through a wrapper.

set -euo pipefail

SINGULARITY_BIN="${SINGULARITY_BIN:-$(command -v singularity || true)}"
if [[ -z "$SINGULARITY_BIN" ]]; then
    echo "Error: singularity is not available on PATH" >&2
    exit 127
fi

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
JUDGE_MODEL_CONFIG="${JUDGE_MODEL_CONFIG:-configs/models/judges/medgemma_27b_it_judge.yaml}"
RUN_CONFIG="${RUN_CONFIG:-configs/runs/full_production.yaml}"

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

mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

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

# Every singularity exec below runs via `srun --export="${singularity_exports}"`,
# which explicitly builds the container's environment from scratch (the
# leading "ALL," plus every SINGULARITYENV_* pair here) - so these values
# only need to exist in this one string, not as separately exported shell
# variables too.
singularity_exports="ALL,SINGULARITYENV_HF_HOME=$HF_HOME,SINGULARITYENV_HUGGINGFACE_HUB_CACHE=$HF_HOME/hub,SINGULARITYENV_HF_HUB_OFFLINE=$HF_HUB_OFFLINE,SINGULARITYENV_TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE,SINGULARITYENV_HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE,SINGULARITYENV_HF_OFFLINE=$HF_OFFLINE,SINGULARITYENV_HF_EVICT_BETWEEN_MODELS=$HF_EVICT_BETWEEN_MODELS,SINGULARITYENV_PYTORCH_ALLOC_CONF=expandable_segments:True,SINGULARITYENV_PYTHONPATH=$WORKDIR${PYTHONPATH:+:$PYTHONPATH},SINGULARITYENV_LLM_API_KEY=${LLM_API_KEY:-EMPTY},SINGULARITYENV_LLM_BENCH_RUNTIME_CONFIG=configs/runtime/telemetry.auto.yaml,SINGULARITYENV_LLM_MAX_TOKENS_DEFAULT=1024,SINGULARITYENV_LLM_NODE_COUNT=$node_count,SINGULARITYENV_SERVE_PORT=$SERVE_PORT,SINGULARITYENV_RUN_CONFIG=$RUN_CONFIG"

mkdir -p logs/orchestration/out logs/orchestration/err logs/vllm outputs/reports outputs/raw outputs/judged

echo "Job $SLURM_JOB_ID on node $(hostname), nodes=${node_count}, gpus_per_node=${gpus_per_node}"

if [[ "$multi_node" -eq 1 ]]; then
    echo "Bootstrapping Ray cluster on allocated nodes..."
    head_node="${nodes_array[0]}"
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" /bin/hostname --ip-address | awk '{print $1}')

    export SINGULARITYENV_VLLM_HOST_IP="$head_node_ip"
    export SINGULARITYENV_LLM_BASE_URL="http://${head_node_ip}:${SERVE_PORT}/v1"

    # Start Ray head
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --export="${singularity_exports},SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,SINGULARITYENV_VLLM_HOST_IP=$head_node_ip,SINGULARITYENV_LLM_BASE_URL=http://${head_node_ip}:${SERVE_PORT}/v1" \
        "$SINGULARITY_BIN" exec --nv --env-file .env "$SIF" \
        ray start --head --node-ip-address="$head_node_ip" --port="$RAY_PORT" --num-gpus="$gpus_per_node" --block &
    RAY_PIDS+=("$!")
    sleep 8

    for ((i=1; i<${#nodes_array[@]}; i++)); do
        worker_node="${nodes_array[$i]}"
        worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" /bin/hostname --ip-address | awk '{print $1}')

        srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" --export="${singularity_exports},SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,SINGULARITYENV_VLLM_HOST_IP=$worker_ip,SINGULARITYENV_LLM_BASE_URL=http://${head_node_ip}:${SERVE_PORT}/v1" \
            "$SINGULARITY_BIN" exec --nv --env-file .env "$SIF" \
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
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" /bin/hostname --ip-address | awk '{print $1}')
    TELEMETRY_ENDPOINTS+=("http://${node_ip}:${telemetry_port}")
    srun --overlap --nodes=1 --ntasks=1 -w "$node" --export="${singularity_exports}" \
        "$SINGULARITY_BIN" exec --nv --env-file .env "$SIF" \
        python3 scripts/telemetry/nvidia_smi_agent.py --host 0.0.0.0 --port "$telemetry_port" --poll-interval-s 0.2 &
    TELEMETRY_PIDS+=("$!")
    sleep 2
done

{
    if [[ ${#TELEMETRY_ENDPOINTS[@]} -eq 0 ]]; then
        echo "telemetry:"
        echo "  enabled: false"
        echo "Telemetry disabled: no endpoints were configured for this run"
    else
        echo "telemetry:"
        echo "  enabled: true"
        echo "  collector: remote_http"
        echo "  timeout_s: 1.5"
        echo "  window_path: /window"
        echo "  endpoints:"
        for endpoint in "${TELEMETRY_ENDPOINTS[@]}"; do
            echo "    - ${endpoint}"
        done
    fi
} > configs/runtime/telemetry.auto.yaml

echo "Telemetry endpoints configured: ${#TELEMETRY_ENDPOINTS[@]} node(s)"
echo "Cache mode: ${HF_CACHE_MODE} | HF_HOME=${HF_HOME} | Offline=${HF_OFFLINE}"
echo "Run config: ${RUN_CONFIG}"

python_script="scripts/orchestration/orchestrator.py"
echo "Running orchestrator: $python_script"

# EXIT_CODE is captured via `|| EXIT_CODE=$?`, not a bare `EXIT_CODE=$?`
# after the command, because under `set -e` a failing plain command aborts
# the script immediately (straight to the cleanup trap) before the next
# line would even run - that would make the success/failure branching below
# unreachable on the failure path.
EXIT_CODE=0
srun --overlap --nodes=1 --ntasks=1 \
    --export="${singularity_exports}" \
    "$SINGULARITY_BIN" exec --nv --env-file .env "$SIF" \
    python3 -u "$python_script" || EXIT_CODE=$?

echo "Orchestrator exited with code: $EXIT_CODE"

if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "Running LLM-as-judge pass: $JUDGE_MODEL_CONFIG"

    JUDGE_EXIT_CODE=0
    srun --overlap --nodes=1 --ntasks=1 \
        --export="${singularity_exports}" \
        "$SINGULARITY_BIN" exec --nv --env-file .env "$SIF" \
        python3 -u scripts/orchestration/llm_judge_run.py --judge-model "$JUDGE_MODEL_CONFIG" || JUDGE_EXIT_CODE=$?

    echo "Judge pass exited with code: $JUDGE_EXIT_CODE"
    EXIT_CODE=$JUDGE_EXIT_CODE
else
    echo "Skipping LLM-as-judge pass: orchestrator did not exit cleanly"
fi

exit $EXIT_CODE
