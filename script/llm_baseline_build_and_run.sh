#!/usr/bin/env bash
# Builds the apptainer container for llm_baseline and submits to SLURM
#
# Usage: ./script/llm_baseline_build_and_run.sh [--local] [--dry-build] [-- <sbatch_flags>] [-- <llm_baseline args>]
# Example: ./script/llm_baseline_build_and_run.sh
# Example: ./script/llm_baseline_build_and_run.sh --local

set -e

LOCAL_MODE=false
DRY_BUILD=false
SBATCH_FLAGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --dry-build)
            DRY_BUILD=true
            shift
            ;;
        --)
            shift
            SBATCH_FLAGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CONTAINER_NAME="llm_baseline.sif"
CONTAINER_PATH=".cache/pokeagent/containers/${CONTAINER_NAME}"

mkdir -p .cache/pokeagent/containers

echo "Building container: ${CONTAINER_NAME}"
echo "This may take several minutes..."

apptainer build --mksquashfs-args "-processors 32" "${CONTAINER_PATH}" ./dconfig/apptainer_llm_baseline.def

echo "Container built successfully: ${CONTAINER_PATH}"

if [[ "$DRY_BUILD" == "true" ]]; then
    echo ""
    echo "Dry-build mode: Skipping execution."
    echo "To run: CONTAINER_NAME=\"${CONTAINER_NAME}\" bash script/llm_baseline.sh ${EXTRA_ARGS[*]}"
    echo ""
elif [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running locally..."
    export CONTAINER_NAME="${CONTAINER_NAME}"
    bash script/llm_baseline.sh "${EXTRA_ARGS[@]}"
else
    echo "Submitting job to SLURM..."
    sbatch \
        --export="CONTAINER_NAME=${CONTAINER_NAME}" \
        "${SBATCH_FLAGS[@]}" \
        script/llm_baseline.sh "${EXTRA_ARGS[@]}"
    echo "Job submitted with container: ${CONTAINER_NAME}"
fi
