#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"

docker run --name agent_run --gpus all "$IMAGE_NAME:$TAG"
docker cp agent_run:/app/emulator/data/output.mp4 ./.cache/output.mp4
docker rm -f agent_run

echo "Agent run complete!"