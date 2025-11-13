#!/bin/bash
set -euo pipefail

DOCKER_TAR="$1"
OUTPUT_SIF="${2:-${DOCKER_TAR%.tar}.sif}"

if ! command -v apptainer &> /dev/null; then
    echo "Apptainer not found. Please install Apptainer to use this script."
    exit 1
fi

if [[ ! -f "$DOCKER_TAR" ]]; then
    echo "Input file '$DOCKER_TAR' does not exist."
    exit 1
fi

echo "Converting Docker tar '$DOCKER_TAR' to Apptainer SIF '$OUTPUT_SIF'..."
apptainer build "$OUTPUT_SIF" "docker-archive://$DOCKER_TAR"

echo "Conversion complete: $OUTPUT_SIF"
