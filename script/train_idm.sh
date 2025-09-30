#!/bin/bash

source .venv/bin/activate
export EXPERIMENT_RUN="1"
export TMPDIR="/scratch/b3schnei/tmp"
NUM_GPUS=${CUDA_VISIBLE_DEVICES:+$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')}
NUM_GPUS=${NUM_GPUS:-8}


# Launch torchrun with 8 processes on a single node
./.venv/bin/torchrun \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=6602 \
  ./train/train_idm.py \
  --config $1