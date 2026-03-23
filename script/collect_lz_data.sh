#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=0
#SBATCH --job-name=lz_collect
#SBATCH --output=/scratch/%u/slurm_out/%j_lz_collect_output.txt
#SBATCH --mail-type=ALL

source .venv/bin/activate
export PYTHONPATH=$(pwd)

ROM=${1:-".cache/lz/rom/lz_rom.gba"}
OUTPUT_DIR=${2:-".cache/lz/rnd_policy"}
STEPS=${3:-300000}
NUM_ROLLOUTS=${4:-1}
NUM_WORKERS=${5:-8}

python script/collect_lz_data.py "$ROM" "$OUTPUT_DIR" "$STEPS" "$NUM_ROLLOUTS" "$NUM_WORKERS"
