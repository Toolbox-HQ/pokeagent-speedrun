#!/usr/bin/env bash
set -e

IMAGE_NAME="pokeagent"
TAG="latest"

# Base bind mounts
BIND_MOUNTS="--bind ./:/app --bind ${HF_HOME:-$HOME/.cache/huggingface}:/hf_cache"

# Conditionally bind /cvmfs if it exists
if [ -d "/cvmfs" ]; then
    BIND_MOUNTS="$BIND_MOUNTS --bind /cvmfs:/cvmfs"
else
    echo "Warning: /cvmfs does not exist, skipping bind."
fi


# Conditionally bind /scratch if it exists
if [ -d "/scratch" ]; then
    BIND_MOUNTS="$BIND_MOUNTS --bind /scratch:/scratch"
else
    echo "Warning: /scratch does not exist, skipping bind."
fi

# Run Apptainer
apptainer exec \
    --contain \
    --nv \
    $BIND_MOUNTS \
    --env PATH="/app/.venv/bin:$PATH" \
    --env HF_HOME=/hf_cache \
    ../test_container.sif \
    bash
