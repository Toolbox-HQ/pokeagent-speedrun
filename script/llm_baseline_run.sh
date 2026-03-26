#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --job-name=llm_baseline
#SBATCH --output=/scratch/%u/slurm_out/%j_llm_baseline_output.txt
#SBATCH --mail-type=ALL

set -e

CONTAINER_NAME="${CONTAINER_NAME:-llm_baseline.sif}"

mkdir -p ./tmp

apptainer exec \
    --contain \
    --nv \
    --bind ./.cache/pokeagent/tmp:/tmp \
    --bind ./.cache:/app/.cache \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --bind ./tmp:/output \
    --env HF_HOME=/hf_cache \
    --env VLLM_CACHE_ROOT=/app/.cache/pokeagent/vllm \
    --env TRITON_HOME="/app/.cache/pokeagent/tmp" \
    --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env PYTHONUNBUFFERED=1 \
    .cache/pokeagent/containers/${CONTAINER_NAME} \
    bash -c "cd /app && . .venv/bin/activate && \
        python llm_baseline.py \
          --rom /app/.cache/pokeagent/rom/rom.gba \
          --save-state /app/.cache/pokeagent/save_state/truck_start.state \
          --video-out /output/llm_baseline_out \
          $*"
