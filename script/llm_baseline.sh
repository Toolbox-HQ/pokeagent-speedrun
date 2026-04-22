#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100
#SBATCH --job-name=llm_baseline
#SBATCH --output=/scratch/%u/slurm_out/%j_llm_baseline_output.txt
#SBATCH --mail-type=ALL

set -e
CONTAINER_NAME="${CONTAINER_NAME:-llm_baseline.sif}"

mkdir -p ./tmp ./.cache/pokeagent/flashinfer

echo "To run via SLURM: sbatch --export=CONTAINER_NAME=${CONTAINER_NAME} script/llm_baseline.sh $*"

EXTRA_ENV=""
if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ENV="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi

apptainer exec \
    --nv \
    --bind ./.cache:/app/.cache \
    --bind ./tmp:/app/tmp \
    --bind ./tmp:/tmp \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --bind ./.cache/pokeagent/flashinfer:"${HOME}/.cache/flashinfer" \
    --env HF_HOME=/hf_cache \
    --env TRITON_HOME="/app/.cache/pokeagent/tmp" \
    --env TRITON_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env TORCHINDUCTOR_CACHE_DIR="/app/.cache/pokeagent/tmp" \
    --env FLASHINFER_CACHE_DIR="/app/.cache/pokeagent/flashinfer" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env PYTHONUNBUFFERED=1 \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    ${EXTRA_ENV:-} \
    .cache/pokeagent/containers/${CONTAINER_NAME} \
    bash -c "cd /app && . .venv/bin/activate && python llm_baseline.py $*"
