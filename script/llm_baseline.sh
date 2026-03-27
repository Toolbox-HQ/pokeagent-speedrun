#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --job-name=llm_baseline
#SBATCH --output=/scratch/%u/slurm_out/%j_llm_baseline_output.txt
#SBATCH --mail-type=ALL

set -e
CONTAINER_NAME="${CONTAINER_NAME:-llm_baseline.sif}"

mkdir -p ./tmp

EXTRA_ENV=""
if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ENV="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi

apptainer exec \
    --contain \
    --nv \
    --bind ./.cache:/app/.cache \
    --bind ./tmp:/app/tmp \
    --bind ./tmp:/tmp \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --env HF_HOME=/hf_cache \
    --env TRITON_HOME="/app/.cache/pokeagent/tmp" \
    --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env TORCHINDUCTOR_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env PYTHONUNBUFFERED=1 \
    ${EXTRA_ENV:-} \
    .cache/pokeagent/containers/${CONTAINER_NAME} \
    bash -c "cd /app && . .venv/bin/activate && python llm_baseline.py $*"
