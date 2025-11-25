from models.dataclass import InferenceArguments
import transformers
from argparse import ArgumentParser

def run_random_agent(conn, steps, uuid):
    from tqdm import tqdm
    from models.inference.random_agent import RandomAgent
    from emulator.keys import KEY_LIST_FOR_IDM
    agent = RandomAgent(30, 180, KEY_LIST_FOR_IDM)
    for i in tqdm(range(steps), desc=f"Random Agent {uuid}"):
        key, num_frames = agent.infer()
        conn.set_key(key)
        conn.run_frames(num_frames)
    conn.close()
    
def run_exploration_agent(start_state, rom_path, output_path, agent_steps, random_steps, interval): 
    from models.inference.agent_inference import Pokeagent
    from tqdm import tqdm
    import torch
    from emulator.emulator_connection import EmulatorConnection
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    import uuid
    
    agent = Pokeagent(device="cuda", temperature=0.01)
    conn = EmulatorConnection(rom_path, output_path + "/output")
    conn.load_state(start_state)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        for i in tqdm(range(agent_steps), desc="Exploration Agent"):
            if i % interval == 0:
                id = str(uuid.uuid4())
                new_conn = EmulatorConnection(rom_path, output_path + f"/{id}")
                new_conn.load_state(conn.get_state())
                executor.submit(run_random_agent, new_conn, random_steps, id)

            tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
            conn.run_frames(7)
            key = agent.infer_action(tensor)
            conn.set_key(key)
            conn.run_frames(23)
        conn.close()

def run_agent(start_state, rom_path, output_path, agent_steps, sampling_strategy, temperature, model_checkpoint, architecture):
    from models.inference.agent_inference import Pokeagent, PokeagentStateOnly
    from tqdm import tqdm
    import torch
    from emulator.emulator_connection import EmulatorConnection
    import numpy as np

    if architecture is "state_only":
        agent = PokeagentStateOnly(model_path=model_checkpoint, device="cuda", temperature=temperature, actions_per_second=4, sampling_strategy=sampling_strategy)
    else:
        raise Exception("only state_only is supported")
    
    conn = EmulatorConnection(rom_path, output_path + "/output")
    conn.load_state(start_state)
    # for i in tqdm(range(agent_steps), desc="Exploration Agent"):
    #     tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
    #     conn.run_frames(7)
    #     key = agent.infer_action(tensor)
    #     conn.set_key(key)
    #     conn.run_frames(23)
    for i in tqdm(range(agent_steps), desc="Exploration Agent"):
        tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
        key = agent.infer_action(tensor)
        conn.run_frames(7)
        conn.set_key(key)
        conn.run_frames(8)
        
    conn.close()    

def main(args: InferenceArguments):
    with open(args.save_state, 'rb') as f:
        state_bytes = f.read()
    AGENT_STEPS = args.agent_steps
    SAMPLING_STRATEGY = args.sampling_strategy
    TEMPERATURE = args.temperature
    MODEL_CHECKPOINT = args.model_checkpoint
    ARCHITECTURE = args.architecture
    RANDOM_STEPS = 1000
    INTERVAL = 20
    ROM_PATH = ".cache/pokeagent/rom/rom.gba"
    OUTPUT_PATH = args.inference_save_path

    #run_exploration_agent(state_bytes, ROM_PATH, OUTPUT_PATH, AGENT_STEPS, RANDOM_STEPS, INTERVAL)
    run_agent(state_bytes, ROM_PATH, OUTPUT_PATH, AGENT_STEPS, SAMPLING_STRATEGY, TEMPERATURE, MODEL_CHECKPOINT, ARCHITECTURE)

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    parser = transformers.HfArgumentParser(InferenceArguments)
    (inference_args,) = parser.parse_yaml_file(yaml_file=args.config)
    main(inference_args)
