#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --job-name=dino_agent_emb
#SBATCH --output=/scratch/bsch/slurmjob_agent_emb_dinov2_%j.txt
#SBATCH --mail-type=FAIL

source p
export PYTHONPATH=$(pwd)
python ./inference/dino_embed.py embed