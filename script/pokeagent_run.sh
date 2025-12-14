#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=pokeagent_online
#SBATCH --output=/scratch/%u/slurm_out/%j_pokeagent_online_output.txt
#SBATCH --mail-type=ALL

set -e

# Container name (defaults to run.sif if not set)
CONTAINER_NAME="${CONTAINER_NAME:-run.sif}"

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
          --nproc_per_node=$NUM_GPUS \
          --nnodes=1 \
          --node_rank=0 \
          --master_addr=localhost \
          --master_port=35332 \
          main.py \
          --config \"$1\""

