#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=08:00:00
#SBATCH --job-name=lz_collect
#SBATCH --output=/scratch/%u/slurm_out/%j_lz_collect_output.txt
#SBATCH --mail-type=ALL

set -e

CONTAINER_NAME="${CONTAINER_NAME:-dev.sif}"

ROM=${1:-.cache/lz/rom/lz_rom.gba}
OUTPUT_DIR=${2:-.cache/lz/rnd_policy}
STEPS=${3:-300000}
N_PER_JOB=${4:-5}

N_WORKERS=${SLURM_NTASKS_PER_NODE:-192}

for i in $(seq 1 $N_WORKERS); do
    apptainer exec \
        --contain \
        --bind ./.cache/pokeagent/tmp:/tmp \
        --bind ./config:/app/config \
        --bind ./dconfig:/app/dconfig \
        --bind ./emulator:/app/emulator \
        --bind ./models:/app/models \
        --bind ./s3_utils:/app/s3_utils \
        --bind ./script:/app/script \
        --bind ./.s3cfg:/app/.s3cfg \
        --bind ./main.py:/app/main.py \
        --bind ./.git:/app/.git \
        --bind ./.cache:/app/.cache \
        --env LZ_MODE=1 \
        .cache/pokeagent/containers/${CONTAINER_NAME} \
        bash -c "cd /app && . .venv/bin/activate && \
            python script/collect_lz_data.py \
              \"${ROM}\" \
              \"${OUTPUT_DIR}\" \
              \"${STEPS}\" \
              \"${N_PER_JOB}\"" &
done
wait
