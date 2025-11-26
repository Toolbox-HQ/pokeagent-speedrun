#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"

INDIVIDUAL_BINDS=" \
    --bind ./config:/app/config \
    --bind ./dconfig:/app/dconfig \
    --bind ./emulator:/app/emulator \
    --bind ./models:/app/models \
    --bind ./s3_utils:/app/s3_utils \
    --bind ./script:/app/script \
    --bind ./.s3cfg:/app/.s3cfg \
    --bind ./main.py:/app/main.py \
"

# Combine all binds
BIND_MOUNTS="$BIND_MOUNTS $INDIVIDUAL_BINDS"

# Build cmd for dev is:
# apptainer build ./.cache/pokeagent/containers/dev.sif ./dconfig/apptainer_dev.def

# Run Apptainer
apptainer run \
    --contain \
    --nv \
    $INDIVIDUAL_BINDS \
    --bind ./.cache/pokeagent/tmp:/tmp \
    --bind ./.cache:/app/.cache \
    --bind "${HF_HOME:-$HOME/.cache/huggingface}":/hf_cache \
    --env HF_HOME=/hf_cache \
    .cache/pokeagent/containers/dev.sif