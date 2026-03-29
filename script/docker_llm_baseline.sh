#!/usr/bin/env bash
# Builds the Docker image for llm_baseline and runs it locally.
# Rebuild is fast when nothing has changed (Docker layer cache).
#
# Usage: ./script/docker_llm_baseline.sh [--gpu <id|all>] [llm_baseline.py args]
# Example: ./script/docker_llm_baseline.sh --steps 1000
# Example: ./script/docker_llm_baseline.sh --gpu 0 --steps 1000
# Example: ./script/docker_llm_baseline.sh --gpu 0,1 --steps 1000

set -e

IMAGE_NAME="llm_baseline"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_VISIBLE_DEVICES=""

PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${PASS_ARGS[@]}"

mkdir -p "${REPO_ROOT}/tmp"

echo "Building Docker image: ${IMAGE_NAME}"
docker build \
    -f "${REPO_ROOT}/dconfig/Dockerfile.llm_baseline" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
echo "Build complete."

EXTRA_ENV=()
if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ENV+=(-e "WANDB_API_KEY=${WANDB_API_KEY}")
fi
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    EXTRA_ENV+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

echo "Launching container..."
docker run --rm \
    --gpus all \
    --ipc=host \
    -v "${REPO_ROOT}/.cache:/app/.cache" \
    -v "${REPO_ROOT}/tmp:/app/tmp" \
    -v "/tmp:/tmp" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/hf_cache" \
    -e HF_HOME=/hf_cache \
    -e TRITON_HOME="/app/.cache/pokeagent/tmp" \
    -e TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    -e TORCHINDUCTOR_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e PYTHONUNBUFFERED=1 \
    "${EXTRA_ENV[@]}" \
    "${IMAGE_NAME}" \
    .venv/bin/python llm_baseline.py "$@"
