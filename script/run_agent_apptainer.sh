#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"

# TODO make it auto run  
apptainer exec \
    --contain \
    --nv \
    --bind ./.cache:/app/.cache \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --env HF_HOME=/hf_cache \
    .cache/pokeagent/containers/pokeagent_latest.sif \
    bash -c "cd /app && . .venv/bin/activate && python main.py