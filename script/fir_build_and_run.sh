#!/usr/bin/env bash
# Builds the apptainer container with a unique name and submits it to SLURM
#
# Usage: ./build_and_run.sh <config_file>
# Example: ./build_and_run.sh config/online/online_agent.yaml

set -e
module load apptainer/1.3.5

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 config/online/online_agent.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Generate unique container name using timestamp
CONTAINER_NAME="run_$(date +%Y%m%d_%H%M%S).sif"
CONTAINER_PATH=".cache/pokeagent/containers/${CONTAINER_NAME}"

# Ensure containers directory exists
mkdir -p .cache/pokeagent/containers

echo "Building container: ${CONTAINER_NAME}"
echo "This may take several minutes..."

# Build the container
apptainer build "${CONTAINER_PATH}" ./dconfig/apptainer_run.def

echo "Container built successfully: ${CONTAINER_PATH}"
echo "Submitting job to SLURM..."

# Submit to SLURM with the container name as an environment variable
sbatch \
    --export=CONTAINER_NAME="${CONTAINER_NAME}" \
    script/fir_pokeagent_run.sh "${CONFIG_FILE}"

echo "Job submitted with container: ${CONTAINER_NAME}"

