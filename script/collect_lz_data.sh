#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=0
#SBATCH --job-name=lz_collect
#SBATCH --output=/scratch/%u/slurm_out/%j_lz_collect_output.txt
#SBATCH --mail-type=ALL

source .venv/bin/activate
export PYTHONPATH=$(pwd)
export LZ_MODE=1

ROM=${1:-"rom/loz.gba"}
OUTPUT_DIR=${2:-".cache/lz/rnd_policy"}
STEPS=${3:-300000}
NUM_ROLLOUTS=${4:-1}

for i in $(seq 1 $NUM_ROLLOUTS); do
    RND=$RANDOM
    python -c "
import sys, random, glob
from emulator.emulator_connection import EmulatorConnection
from models.inference.random_agent import RandomAgent
from emulator.keys import KEY_LIST_FOR_IDM

rom = sys.argv[1]
output_dir = sys.argv[2]
steps = int(sys.argv[3])
rnd = sys.argv[4]

conn = EmulatorConnection(rom)
states = glob.glob('.cache/lz/state/*.state')
if states:
    conn.load_state_from_file(random.choice(states))

agent = RandomAgent(30, 180, KEY_LIST_FOR_IDM)
conn.create_video_writer(f'{output_dir}/rollout_{rnd}')
conn.start_video_writer(f'{output_dir}/rollout_{rnd}')
i = 0
while i < steps:
    key, num_frames = agent.infer()
    conn.set_key(key)
    conn.run_frames(num_frames)
    i += num_frames
conn.release_video_writer(f'{output_dir}/rollout_{rnd}')
conn.close()
" "$ROM" "$OUTPUT_DIR" "$STEPS" "$RND"
done
