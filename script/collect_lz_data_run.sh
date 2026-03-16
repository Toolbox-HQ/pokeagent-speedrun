#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=0
#SBATCH --job-name=lz_collect
#SBATCH --output=/scratch/%u/slurm_out/%j_lz_collect_output.txt
#SBATCH --mail-type=ALL

set -e

CONTAINER_NAME="${CONTAINER_NAME:-run.sif}"

ROM=${1:-.cache/lz/rom/lz_rom.gba}
OUTPUT_DIR=${2:-.cache/lz/rnd_policy}
STEPS=${3:-300000}
NUM_ROLLOUTS=${4:-1}

apptainer exec \
    --contain \
    --bind ./.cache:/app/.cache \
    .cache/pokeagent/containers/${CONTAINER_NAME} \
    bash -c "cd /app && . .venv/bin/activate && \
        bash script/collect_lz_data.sh \
          \"${ROM}\" \
          \"${OUTPUT_DIR}\" \
          \"${STEPS}\" \
          \"${NUM_ROLLOUTS}\""
