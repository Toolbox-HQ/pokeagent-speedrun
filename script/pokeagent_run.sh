#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=pokeagent_online
#SBATCH --output=/scratch/%u/slurm_out/%j_pokeagent_online_output.txt
#SBATCH --mail-type=ALL

set -e
# Container name (defaults to run.sif if not set)
CONTAINER_NAME="${CONTAINER_NAME:-run.sif}"

# Run UUID (defaults to container base name without .sif if not set)
RUN_UUID="${RUN_UUID:-${CONTAINER_NAME%.sif}}"

# Multi-node coordination — works for both single- and multi-node jobs
NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=35332
CONFIG_ARG="$1"

EXTRA_ENV_ARGS=""
if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ENV_ARGS="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi
if [[ -n "${LZ_MODE}" ]]; then
    EXTRA_ENV_ARGS="${EXTRA_ENV_ARGS} --env LZ_MODE=${LZ_MODE}"
fi

# Export so each srun task inherits these values
export NNODES MASTER_ADDR MASTER_PORT RUN_UUID CONTAINER_NAME CONFIG_ARG EXTRA_ENV_ARGS

# srun launches one task per node; SLURM_NODEID is set per-task by SLURM
srun --ntasks-per-node=1 bash -c '
    # Determine GPUs visible on this node
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
    else
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    fi

    apptainer exec \
        --contain \
        --nv \
        --bind ./.cache/pokeagent/tmp:/tmp \
        --bind ./.cache:/app/.cache \
        --bind ./checkpoints:/app/checkpoints \
        --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
        --env HF_HOME=/hf_cache \
        --env TRITON_HOME="/app/.cache/pokeagent/tmp" \
        --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
        --env WANDB_MODE="offline" \
        --env PYTHONUNBUFFERED=1 \
        --bind /dev/infiniband \
        ${EXTRA_ENV_ARGS} \
        .cache/pokeagent/containers/${CONTAINER_NAME} \
        bash -c "cd /app && . .venv/bin/activate && \
            echo '=== Node ${SLURM_NODEID}: GPUs ===' && \
            nvidia-smi -L && \
            echo '=== Node ${SLURM_NODEID}: IB devices ===' && \
            ibv_devinfo && \
            echo '=== Node ${SLURM_NODEID}: launching torchrun ===' && \
            NCCL_DEBUG=VERSION \
            torchrun \
              --nproc_per_node=${NUM_GPUS} \
              --nnodes=${NNODES} \
              --node_rank=${SLURM_NODEID} \
              --master_addr=${MASTER_ADDR} \
              --master_port=${MASTER_PORT} \
              main.py \
              --config \"${CONFIG_ARG}\" \
              --uuid \"${RUN_UUID}\" \
              --seed_rng \"false\""
'

