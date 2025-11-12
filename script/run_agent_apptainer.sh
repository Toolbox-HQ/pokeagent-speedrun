#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

apptainer exec \
    --contain \
    --nv \
    --bind ./.cache:/app/.cache \
    --bind "$HF_DIR":/hf_cache \
    --env HF_HOME=/hf_cache \
    ../pokeagent.sif \
    "cd /app && bash ./script/run_agent.sh"
