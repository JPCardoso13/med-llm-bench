#!/bin/bash
#SBATCH --job-name=vllm_multinode
#SBATCH -p rtx4060
#SBATCH -A haslab
#SBATCH --nodes=2
#SBATCH --nodelist=aurora[06-07]
#SBATCH --time=02:00:00
#SBATCH --output=logs/vllm_%j.log

# 1. Load variables
export HF_HOME=/projects/jcardoso/med-llm-bench/.cache/huggingface
export SIF="med-llm-bench.sif"

# 2. Identify the Head Node IP address
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379

echo "Starting Ray Head on $head_node ($head_node_ip)"

# 3. Start the Ray Head on the first node
srun --nodes=1 --ntasks=1 -w "$head_node" \
    env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP=$head_node_ip \
    singularity exec --nv $SIF \
    ray start --head --node-ip-address=$head_node_ip --port=$port --num-gpus=1 --block &

sleep 10 # Wait for head to initialize

# 4. Start the Ray Worker on the second node
worker_node=${nodes_array[1]}
worker_ip=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address)
echo "Starting Ray Worker on $worker_node ($worker_ip)"

srun --nodes=1 --ntasks=1 -w "$worker_node" \
    env SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_VLLM_HOST_IP=$worker_ip \
    singularity exec --nv $SIF \
    ray start --address="$head_node_ip:$port" --node-ip-address=$worker_ip --num-gpus=1 --block &

sleep 10 # Wait for worker to connect

# 5. Connect Python to the Ray cluster and run the script
export RAY_ADDRESS="${head_node_ip}:${port}"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_VLLM_HOST_IP=$head_node_ip
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "Starting vLLM script..."
singularity exec --nv --env-file .env $SIF \
    python3 -u scripts/dataset_creation/medcasereasoning/label_specialties.py --limit 20
