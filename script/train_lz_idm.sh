#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --job-name=lz_idm_job
#SBATCH --output=/scratch/%u/slurm_out/%j_lz_idm_job_output.txt
#SBATCH --mail-type=ALL

source .venv/bin/activate
export PYTHONPATH=$(pwd)
export EXPERIMENT_RUN="1"
export TMPDIR="/scratch/b3schnei/tmp"
export WANDB_MODE="offline"
export LZ_MODE=1
NUM_GPUS=${CUDA_VISIBLE_DEVICES:+$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')}
NUM_GPUS=${NUM_GPUS:-4}

./.venv/bin/torchrun \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=6602 \
  ./models/train/train_idm.py \
  --config $1
