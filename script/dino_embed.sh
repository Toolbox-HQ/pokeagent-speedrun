#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --job-name=dino_agent_emb
#SBATCH --output=/scratch/%u/slurm_out/%j_dinov2_inference_job_output.txt
#SBATCH --mail-type=ALL

source p
export PYTHONPATH=$(pwd)
python ../models/inference/dino_embed.py embed