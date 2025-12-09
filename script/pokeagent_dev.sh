#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --job-name=pokeagent_online
#SBATCH --output=/scratch/%u/slurm_out/%j_pokeagent_online_output.txt
#SBATCH --mail-type=ALL

set -e
INDIVIDUAL_BINDS=" \
    --bind ./config:/app/config \
    --bind ./dconfig:/app/dconfig \
    --bind ./emulator:/app/emulator \
    --bind ./models:/app/models \
    --bind ./s3_utils:/app/s3_utils \
    --bind ./script:/app/script \
    --bind ./.s3cfg:/app/.s3cfg \
    --bind ./main.py:/app/main.py \
    --bind ./.git:/app/.git
"

# Combine all binds
BIND_MOUNTS="$BIND_MOUNTS $INDIVIDUAL_BINDS"

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
    $INDIVIDUAL_BINDS \
    --bind ./.cache/pokeagent/tmp:/tmp \
    --bind ./.cache:/app/.cache \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --env HF_HOME=/hf_cache \
    --env TRITON_HOME="/app/.cache/pokeagent/tmp" \
    --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env WANDB_MODE="offline" \
    ${EXTRA_ENV:-} \
    .cache/pokeagent/containers/dev.sif \
    bash -c "cd /app && . .venv/bin/activate && \
        torchrun \
          --nproc_per_node=$NUM_GPUS \
          --nnodes=1 \
          --node_rank=0 \
          --master_addr=localhost \
          --master_port=35332 \
          main.py \
          --config \"$1\""
