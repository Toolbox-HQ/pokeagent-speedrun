# from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments
# import transformers
# from argparse import ArgumentParser
# from models.inference.agent_inference import Pokeagent, PokeagentStateOnly, PokeAgentActionConditioned
# from tqdm import tqdm
# import torch
# from emulator.emulator_connection import EmulatorConnection
# import numpy as np
# from models.inference.random_agent import RandomAgent

# from models.inference.agent_inference import Pokeagent
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# import uuid
# from models.util.repro import repro_init
# from models.train.train_idm import IDMArguments
# from argparse import ArgumentParser

def run_random_agent(conn, steps, uuid):
    from models.inference.random_agent import RandomAgent
    from emulator.keys import KEY_LIST_FOR_IDM
    agent = RandomAgent(30, 180, KEY_LIST_FOR_IDM)
    for i in range(steps):
        key, num_frames = agent.infer()
        conn.set_key(key)
        conn.run_frames(num_frames)
    conn.close()
    
def run_online_agent(model_args, data_args, training_args, inference_args, idm_args): 
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

def run_agent(inference_architecture, model_checkpoint, agent_steps, save_state, sampling_strategy,  temperature, actions_per_second, inference_save_path, agent_fps, context_length, online, rom_path, idm_data_sample_interval, idm_data_sample_length, agent_data_sample_length):
    from models.inference.agent_inference import PokeagentStateOnly, PokeAgentActionConditioned
    from emulator.emulator_connection import EmulatorConnection
    from tqdm import tqdm
    import numpy as np
    import torch
    import uuid
    
    with open(save_state, 'rb') as f:
        state_bytes = f.read()
    
    if inference_architecture == "state_only":
        agent = PokeagentStateOnly(model_path=model_checkpoint, device="cuda", temperature=temperature, actions_per_second=4, sampling_strategy=sampling_strategy)
    elif inference_architecture == "state_action_conditioned":
        agent = PokeAgentActionConditioned(model_path=model_checkpoint, device="cuda", temperature=temperature, actions_per_second=4, sampling_strategy=sampling_strategy)
    else:
        raise Exception(f"{inference_architecture} is not supported")
    
    video_path = inference_save_path + f'/{str(uuid.uuid4())}'
    conn = EmulatorConnection(rom_path)
    conn.load_state(state_bytes)
    conn.create_video_writer(video_path)
    conn.start_video_writer(video_path)
    for i in tqdm(range(agent_steps), desc="Exploration Agent"):
        tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
        key = agent.infer_action(tensor)
        conn.run_frames(7)
        conn.set_key(key)
        conn.run_frames(8)
    conn.release_video_writer(video_path)
    conn.close()

def main(model_args, data_args, training_args, inference_args, idm_args):
   
    if inference_args.online:
        run_online_agent(model_args, data_args, training_args, inference_args, idm_args)
    else:
        run_agent(*inference_args)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import transformers
    from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments
    from models.train.train_idm import IDMArguments

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    #save_path = repro_init(args.config)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, InferenceArguments, IDMArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        inference_args,
        idm_args
    ) = parser.parse_yaml_file(yaml_file=args.config)

    main(model_args,
        data_args,
        training_args,
        inference_args,
        idm_args)
