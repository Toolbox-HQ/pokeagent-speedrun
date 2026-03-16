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

python -c "
import random, glob, uuid, os
from joblib import Parallel, delayed
from emulator.emulator_connection import EmulatorConnection
from models.inference.random_agent import LZRandomAgent
from emulator.keys import KEY_LIST_FOR_IDM

def run_rollout(rom, output_dir, steps):
    rnd = uuid.uuid4().hex[:8]
    conn = EmulatorConnection(rom)
    states = glob.glob('.cache/lz/state/*.state')
    if states:
        conn.load_state_from_file(random.choice(states))
    agent = LZRandomAgent(30, 180, KEY_LIST_FOR_IDM)
    os.makedirs(output_dir, exist_ok=True)
    conn.create_video_writer(f'{output_dir}/output_{rnd}')
    conn.start_video_writer(f'{output_dir}/output_{rnd}')
    i = 0
    while i < steps:
        key, num_frames = agent.infer()
        conn.set_key(key)
        conn.run_frames(num_frames)
        i += num_frames
    conn.release_video_writer(f'{output_dir}/output_{rnd}')
    conn.close()
    os.rename(f'{output_dir}/output_{rnd}.json', f'{output_dir}/keys_{rnd}.json')

import sys, os
rom, output_dir, steps, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
n_jobs = min(n, os.cpu_count())
Parallel(n_jobs=n_jobs)(delayed(run_rollout)(rom, output_dir, steps) for _ in range(n))
" "$ROM" "$OUTPUT_DIR" "$STEPS" "$NUM_ROLLOUTS"
