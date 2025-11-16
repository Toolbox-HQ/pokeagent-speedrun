#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --job-name=agent_job
#SBATCH --output=/scratch/bsch/slurm_out/%j_agent_job_output.txt
#SBATCH --mail-type=ALL

cd /scratch/bsch/pokeagent-speedrun
source .venv/bin/activate

# Environment variables
export EXPERIMENT_RUN="1"
export WANDB_MODE="offline"
export PYTHONPATH=$(pwd)
export WANDB_DIR="./wandb"

export WORK="/scratch/bsch"
export CUDA_HOME="$WORK/anaconda3/envs/cuda"
export HF_HOME="$WORK/hf_cache"
export APPTAINER_CACHEDIR="$WORK/.apptainer"
export TRITON_CACHE_DIR="$WORK/.triton"

# Automatically detect all available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

for p in $(seq 6600 6900); do
    if ! lsof -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1; then
        MASTER_PORT=$p
        break
    fi
done

if [ -z "$MASTER_PORT" ]; then
    echo "No free port found" >&2
    exit 1
fi

# Run torchrun using all GPUs on this node
./.venv/bin/torchrun \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=$MASTER_PORT \
  ./models/train/train_agent.py \
  --config $1