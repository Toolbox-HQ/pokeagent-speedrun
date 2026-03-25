#!/bin/bash
#SBATCH --gpus=h100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --job-name=pokeagent_online
#SBATCH --output=/scratch/%u/slurm_out/%j_pokeagent_online_output.txt
#SBATCH --mail-type=ALL

set -e
module load apptainer/1.3.5

# Container name (defaults to run.sif if not set)
CONTAINER_NAME="${CONTAINER_NAME:-run.sif}"

# Run UUID (defaults to container base name without .sif if not set)
RUN_UUID="${RUN_UUID:-${CONTAINER_NAME%.sif}}"

# Determine number of GPUs
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ENV="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi

apptainer exec \
    --contain \
    --nv \
    --bind ./.cache/pokeagent/tmp:/tmp \
    --bind ./.cache:/app/.cache \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --env HF_HOME=/hf_cache \
    --env TRITON_HOME="/app/.cache/pokeagent/tmp" \
    --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env WANDB_MODE="offline" \
    ${EXTRA_ENV:-} \
    .cache/pokeagent/containers/${CONTAINER_NAME} \
    bash -c "cd /app && . .venv/bin/activate && \
        torchrun \
            --nproc_per_node=$SLURM_GPUS_ON_NODE \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --master_addr=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1) \
            --master_port=29500 \
            main.py \
            --config \"$1\" \
            --uuid \"${RUN_UUID}\""

