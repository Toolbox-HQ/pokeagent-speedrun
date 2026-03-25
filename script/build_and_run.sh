#!/usr/bin/env bash
# Builds the apptainer container with a unique name and submits it to SLURM
#
# Usage: ./build_and_run.sh [--local] [--dry-build] <config_file> [-- <sbatch_flags>]
# Example: ./build_and_run.sh config/online/online_agent.yaml
# Example: ./build_and_run.sh --local config/online/online_agent.yaml
# Example: ./build_and_run.sh --dry-build config/online/online_agent.yaml
# Example: ./build_and_run.sh config/online/online_agent.yaml -- --time=2:00:00 --mem=16G

set -e

# Parse arguments
LOCAL_MODE=false
DRY_BUILD=false
LZ_MODE=false
CONFIG_FILE=""
SBATCH_FLAGS=()

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
        --lz)
            LZ_MODE=true
            shift
            ;;
        --)
            # Everything after -- goes to sbatch
            shift
            SBATCH_FLAGS=("$@")
            break
            ;;
        *)
            if [[ -z "$CONFIG_FILE" ]]; then
                CONFIG_FILE="$1"
            else
                echo "Error: Multiple config files specified"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: $0 [--local] [--dry-build] [--lz] <config_file> [-- <sbatch_flags>]"
    echo "Example: $0 config/online/online_agent.yaml"
    echo "Example: $0 --lz config/online/online_agent.yaml"
    echo "Example: $0 --local config/online/online_agent.yaml"
    echo "Example: $0 --dry-build config/online/online_agent.yaml"
    echo "Example: $0 config/online/online_agent.yaml -- --time=2:00:00 --mem=16G"
    exit 1
fi

# Generate a UUID for this run
if command -v uuidgen >/dev/null 2>&1; then
    RUN_UUID="$(uuidgen)"
else
    # Fallback to timestamp-based ID if uuidgen is unavailable
    RUN_UUID="run_$(date +%Y%m%d_%H%M%S)"
fi

# Use the run UUID as the container base name
CONTAINER_NAME="${RUN_UUID}.sif"
CONTAINER_PATH=".cache/pokeagent/containers/${CONTAINER_NAME}"

# Ensure containers directory exists
mkdir -p .cache/pokeagent/containers

echo "Building container: ${CONTAINER_NAME}"
echo "This may take several minutes..."

# Build the container
apptainer build --mksquashfs-args "-processors 32" "${CONTAINER_PATH}" ./dconfig/apptainer_run.def

echo "Container built successfully: ${CONTAINER_PATH}"

if [[ "$DRY_BUILD" == "true" ]]; then
    LZ_ENV=""
    [[ "$LZ_MODE" == "true" ]] && LZ_ENV="LZ_MODE=1 "
    echo ""
    echo "Dry-build mode: Skipping execution."
    echo "To run this container, use one of the following commands:"
    echo ""
    if [[ "$LOCAL_MODE" == "true" ]]; then
        echo "  Local mode:"
        echo "    CONTAINER_NAME=\"${CONTAINER_NAME}\" RUN_UUID=\"${RUN_UUID}\" ${LZ_ENV}bash script/pokeagent_run.sh \"${CONFIG_FILE}\""
    else
        echo "  SLURM mode:"
        DRY_EXPORT="CONTAINER_NAME=${CONTAINER_NAME},RUN_UUID=${RUN_UUID}"
        [[ "$LZ_MODE" == "true" ]] && DRY_EXPORT="${DRY_EXPORT},LZ_MODE=1"
        SBATCH_CMD="sbatch --export=${DRY_EXPORT}"
        if [[ ${#SBATCH_FLAGS[@]} -gt 0 ]]; then
            SBATCH_CMD="${SBATCH_CMD} ${SBATCH_FLAGS[*]}"
        fi
        SBATCH_CMD="${SBATCH_CMD} script/pokeagent_run.sh \"${CONFIG_FILE}\""
        echo "    ${SBATCH_CMD}"
        echo ""
        echo "  Local mode:"
        echo "    CONTAINER_NAME=\"${CONTAINER_NAME}\" RUN_UUID=\"${RUN_UUID}\" ${LZ_ENV}bash script/pokeagent_run.sh \"${CONFIG_FILE}\""
    fi
    echo ""
elif [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running locally (not submitting to SLURM)..."
    export CONTAINER_NAME="${CONTAINER_NAME}"
    export RUN_UUID="${RUN_UUID}"
    [[ "$LZ_MODE" == "true" ]] && export LZ_MODE=1
    bash script/pokeagent_run.sh "${CONFIG_FILE}"
else
    echo "Submitting job to SLURM..."
    EXPORT_VARS="CONTAINER_NAME=${CONTAINER_NAME},RUN_UUID=${RUN_UUID}"
    [[ "$LZ_MODE" == "true" ]] && EXPORT_VARS="${EXPORT_VARS},LZ_MODE=1"
    sbatch \
        --export="${EXPORT_VARS}" \
        "${SBATCH_FLAGS[@]}" \
        script/pokeagent_run.sh "${CONFIG_FILE}"
    echo "Job submitted with container: ${CONTAINER_NAME}"
fi

