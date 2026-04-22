#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --job-name=validate_idm
#SBATCH --output=/scratch/%u/slurm_out/%j_validate_idm_output.txt
#SBATCH --mail-type=ALL

set -e
INDIVIDUAL_BINDS=" \
    --bind ./config:/app/config \
    --bind ./dconfig:/app/dconfig \
    --bind ./emulator:/app/emulator \
    --bind ./models:/app/models \
    --bind ./s3_utils:/app/s3_utils \
    --bind ./script:/app/script \
    --bind ./idm_validate:/app/idm_validate \
    --bind ./main.py:/app/main.py \
    --bind ./.git:/app/.git
"

EXTRA_ENV=""
if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ENV="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    EXTRA_ENV="${EXTRA_ENV} --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env WANDB_MODE="offline" \
    --env PYTHONUNBUFFERED=1 \
    ${EXTRA_ENV:-} \
    .cache/pokeagent/containers/dev.sif \
    bash -c "cd /app && . .venv/bin/activate && \
        python -m idm_validate.validate_idm \
          --config \"$1\"
        "
