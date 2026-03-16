import random, glob, uuid, os, multiprocessing, sys
from functools import partial
from emulator.emulator_connection import EmulatorConnection
from models.inference.random_agent import LZRandomAgent
from emulator.keys import KEY_LIST_FOR_IDM

def run_rollout(rom, output_dir, steps, _=None):
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

if __name__ == '__main__':
    rom, output_dir, steps, n, n_workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    _rollout = partial(run_rollout, rom, output_dir, steps)
    procs = []
    for i in range(n):
        p = multiprocessing.Process(target=_rollout)
        p.start()
        procs.append(p)
        if len(procs) >= n_workers:
            procs.pop(0).join()
    for p in procs:
        p.join()
