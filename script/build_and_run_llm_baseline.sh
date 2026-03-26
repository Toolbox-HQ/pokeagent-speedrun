#!/usr/bin/env bash
# Builds the apptainer container for the LLM baseline and submits to SLURM (or runs locally).
#
# Usage: ./build_and_run_llm_baseline.sh [--local] [--dry-build] [-- <llm_baseline.py args>]
# Example: ./build_and_run_llm_baseline.sh
# Example: ./build_and_run_llm_baseline.sh --local
# Example: ./build_and_run_llm_baseline.sh -- --steps 500 --model Qwen/Qwen3-VL-8B-Instruct
# Example: ./build_and_run_llm_baseline.sh -- --time=4:00:00 --steps 2000

set -e

LOCAL_MODE=false
DRY_BUILD=false
PASS_ARGS=()

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
            PASS_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--local] [--dry-build] [-- <llm_baseline.py args>]"
            exit 1
            ;;
    esac
done

# Generate a unique container name
if command -v uuidgen >/dev/null 2>&1; then
    RUN_UUID="$(uuidgen)"
else
    RUN_UUID="run_$(date +%Y%m%d_%H%M%S)"
fi

CONTAINER_NAME="${RUN_UUID}.sif"
CONTAINER_PATH=".cache/pokeagent/containers/${CONTAINER_NAME}"

mkdir -p .cache/pokeagent/containers

echo "Building container: ${CONTAINER_NAME}"
echo "This may take several minutes..."

apptainer build --mksquashfs-args "-processors 32" "${CONTAINER_PATH}" ./dconfig/apptainer_llm_baseline.def

echo "Container built successfully: ${CONTAINER_PATH}"

if [[ "$DRY_BUILD" == "true" ]]; then
    echo ""
    echo "Dry-build mode: Skipping execution."
    echo "To run this container:"
    echo ""
    echo "  SLURM mode:"
    echo "    CONTAINER_NAME=\"${CONTAINER_NAME}\" sbatch --export=CONTAINER_NAME=${CONTAINER_NAME} script/llm_baseline_run.sh ${PASS_ARGS[*]}"
    echo ""
    echo "  Local mode:"
    echo "    CONTAINER_NAME=\"${CONTAINER_NAME}\" bash script/llm_baseline_run.sh ${PASS_ARGS[*]}"
    echo ""
elif [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running locally (not submitting to SLURM)..."
    export CONTAINER_NAME="${CONTAINER_NAME}"
    bash script/llm_baseline_run.sh "${PASS_ARGS[@]}"
else
    echo "Submitting job to SLURM..."
    sbatch \
        --export="CONTAINER_NAME=${CONTAINER_NAME}" \
        script/llm_baseline_run.sh "${PASS_ARGS[@]}"
    echo "Job submitted with container: ${CONTAINER_NAME}"
fi
