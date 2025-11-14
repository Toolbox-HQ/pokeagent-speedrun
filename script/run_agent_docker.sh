#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --mount type=bind,src=./.cache,dst=/app/.cache --mount type=bind,src=$HF_DIR,dst=/hf_cache --gpus all -e HF_HOME=/hf_cache "$IMAGE_NAME:$TAG"

echo "Agent run complete!"