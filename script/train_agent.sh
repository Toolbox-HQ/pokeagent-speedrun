#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=pretrain_agent_job
#SBATCH --output=/scratch/%u/slurm_out/%j_agent_job_output.txt
#SBATCH --mail-type=ALL

source .venv/bin/activate

# Environment variables
export EXPERIMENT_RUN="1"
export WANDB_MODE="offline"
export PYTHONPATH=$(pwd)
export WANDB_DIR="./"
export CUPY_CACHE_DIR="./.cache/pokeagent/cupy"

#export WORK="/scratch/bsch"
#export CUDA_HOME="$WORK/anaconda3/envs/cuda"
#export APPTAINER_CACHEDIR="$WORK/.apptainer"
export TRITON_CACHE_DIR="./.triton"

# Automatically detect all available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Run torchrun using all GPUs on this node
./.venv/bin/torchrun \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --node_rank=0 \
  --master_port=30999 \
  --master_addr=localhost \
  ./models/train/train_agent.py \
  --config $1