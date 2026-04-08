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

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
serve_port=8000

echo "Starting Ray Head on $head_node ($head_node_ip)"

srun --nodes=1 --ntasks=1 -w "$head_node" \
    env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP=$head_node_ip \
    singularity exec --nv $SIF \
    ray start --head --node-ip-address=$head_node_ip --port=$port --num-gpus=1 --block &
sleep 10

worker_node=${nodes_array[1]}
worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address)
echo "Starting Ray Worker on $worker_node ($worker_ip)"

srun --nodes=1 --ntasks=1 -w "$worker_node" \
    env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP=$worker_ip \
    singularity exec --nv $SIF \
    ray start --address="$head_node_ip:$port" --node-ip-address=$worker_ip --num-gpus=1 --block &
sleep 10

export RAY_ADDRESS="${head_node_ip}:${port}"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_VLLM_HOST_IP=$head_node_ip
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "Starting vLLM server on ${head_node_ip}:${serve_port}..."

singularity exec --nv --env-file .env $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-32B-AWQ \
        --tensor-parallel-size 2 \
        --distributed-executor-backend ray \
        --host 0.0.0.0 \
        --port $serve_port \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.80 \
        --enforce-eager