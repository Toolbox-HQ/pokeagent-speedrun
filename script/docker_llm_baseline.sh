#!/bin/bash
# Build and run the LLM baseline in a Docker container.
# Video output is written to ./tmp/llm_baseline_out on the host (owned by current user).
#
# Usage: bash script/docker_llm_baseline.sh [--gpu <id>] [llm_baseline.py args]
# Example: bash script/docker_llm_baseline.sh --steps 200
# Example: bash script/docker_llm_baseline.sh --gpu 1 --steps 200

set -e

IMAGE_NAME="llm_baseline"
OUTPUT_DIR="$(pwd)/tmp"
GPU_ID="all"

# Parse --gpu flag before passing remaining args to llm_baseline.py
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="device=$2"
            shift 2
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -f dconfig/Dockerfile.llm_baseline -t "${IMAGE_NAME}" .

echo "Running LLM baseline on GPU=${GPU_ID} (output -> ${OUTPUT_DIR})"
docker run --rm \
    --gpus "${GPU_ID}" \
    -v "$(pwd)/.cache:/app/.cache" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/hf_cache" \
    -v "${OUTPUT_DIR}:/output" \
    -e HF_HOME=/hf_cache \
    -e VLLM_CACHE_ROOT=/app/.cache/pokeagent/vllm \
    -e TRITON_HOME=/app/.cache/pokeagent/tmp \
    -e TRITON_CACHE_DIR=/app/.cache/pokeagent/tmp \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -e PYTHONUNBUFFERED=1 \
    "${IMAGE_NAME}" \
    bash -c ".venv/bin/python llm_baseline.py \
        --rom /app/.cache/pokeagent/rom/rom.gba \
        --save-state /app/.cache/pokeagent/save_state/truck_start.state \
        --video-out /output/llm_baseline_out \
        ${PASS_ARGS[*]} && chown -R $(id -u):$(id -g) /output"
