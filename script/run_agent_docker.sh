#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --mount type=bind,src=./.cache,dst=/app/.cache --mount type=bind,src=$HF_DIR,dst=/hf_cache --gpus all -e HF_HOME=/hf_cache "$IMAGE_NAME:$TAG" 
docker cp agent_run:/app/emulator/data/output.mp4 ./.cache/output.mp4
docker rm -f agent_run

echo "Agent run complete!"