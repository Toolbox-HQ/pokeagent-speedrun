def run_random_agent(conn, steps, video_path):
    from models.inference.random_agent import RandomAgent
    from emulator.keys import KEY_LIST_FOR_IDM
    agent = RandomAgent(30, 180, KEY_LIST_FOR_IDM)
    conn.create_video_writer(video_path)
    conn.start_video_writer(video_path)
    for i in range(steps):
        key, num_frames = agent.infer()
        conn.set_key(key)
        conn.run_frames(num_frames)
    conn.release_video_writer(video_path)
    conn.close()
    
def run_online_agent(model_args, data_args, training_args, inference_args, idm_args): 
    from models.inference.agent_inference import OnlinePokeagentStateOnly, OnlinePokeagentStateActionConditioned
    from emulator.emulator_connection import EmulatorConnection
    from tqdm import tqdm
    import numpy as np
    import torch
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    from models.inference.find_matching_video_intervals import get_intervals
    import json
    
    with open(inference_args.save_state, 'rb') as f:
        curr_state = f.read()
    
    if inference_args.inference_architecture == "state_only":
        agent = OnlinePokeagentStateOnly(model_args, training_args, data_args, inference_args, idm_args)
    elif inference_args.inference_architecture == "state_action_conditioned":
        agent = OnlinePokeagentStateActionConditioned(model_args, training_args, data_args, inference_args, idm_args)
    else:
        raise Exception(f"{inference_args.inference_architecture} is not supported")
    

    start = True
    video_path = inference_args.inference_save_path + f'/output'
    bootstap_count = 0
    query_path_template = '.cache/pokeagent/query_video/query'
    query_path = query_path_template + str(bootstap_count)
    conn = EmulatorConnection(inference_args.rom_path)
    conn.load_state(curr_state)
    conn.create_video_writer(query_path)
    conn.start_video_writer(query_path)
    conn.create_video_writer(video_path)
    conn.start_video_writer(video_path)
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in tqdm(range(inference_args.agent_steps), desc="Exploration Agent"):
            if i % inference_args.bootstrap_interval == 0:
                if not start:
                    conn.release_video_writer(query_path)
                    agent.train_idm(".cache/pokeagent/idm_data")
                    video_intervals = get_intervals(query_path, ".cache/pokeagent/dinov2", 540, 400)
                    interval_path = f".cache/pokeagent/agent_data/intervals{bootstap_count}.json"
                    with open(interval_path, "w") as f:
                        json.dump(video_intervals, f)
                    agent.train_agent(interval_path)
                    bootstap_count += 1
                    query_path = query_path_template + str(bootstap_count)
                    conn.create_video_writer(query_path)
                    conn.start_video_writer(query_path)
                else:
                    start = False
            if i % inference_args.idm_data_sample_interval == 0:
                id = str(uuid.uuid4())
                new_conn = EmulatorConnection(inference_args.rom_path)
                new_conn.load_state(conn.get_state())
                executor.submit(run_random_agent, new_conn, inference_args.idm_data_sample_steps, f".cache/pokeagent/idm_data/{id}")
            tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
            key = agent.infer_action(tensor)
            conn.run_frames(7)
            conn.set_key(key)
            conn.run_frames(8)
    conn.release_video_writer(video_path)
    conn.close()

def run_agent(inference_architecture, model_checkpoint, agent_steps, save_state, sampling_strategy,  temperature, actions_per_second, inference_save_path, agent_fps, context_length, online, rom_path, idm_data_sample_interval, idm_data_sample_steps, bootstrap_interval):
    from models.inference.agent_inference import PokeagentStateOnly, PokeAgentActionConditioned
    from emulator.emulator_connection import EmulatorConnection
    from tqdm import tqdm
    import numpy as np
    import torch
    import uuid
    
    with open(save_state, 'rb') as f:
        state_bytes = f.read()
    
    if inference_architecture == "state_only":
        agent = PokeagentStateOnly(model_path=model_checkpoint, device="cuda", temperature=temperature, actions_per_second=actions_per_second, sampling_strategy=sampling_strategy, context_len=context_length, model_fps=agent_fps)
    elif inference_architecture == "state_action_conditioned":
        agent = PokeAgentActionConditioned(model_path=model_checkpoint, device="cuda", temperature=temperature, actions_per_second=actions_per_second, sampling_strategy=sampling_strategy, context_len=context_length, model_fps=agent_fps)
    else:
        raise Exception(f"{inference_architecture} is not supported")
    
    video_path = inference_save_path + f'/{str(uuid.uuid4())}'
    conn = EmulatorConnection(rom_path)
    conn.load_state(state_bytes)
    conn.create_video_writer(video_path)
    conn.start_video_writer(video_path)
    for i in tqdm(range(agent_steps), desc="Inference Agent"):
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
    from models.util.repro import repro_init
    from models.util.dist import init_distributed

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    save_path = repro_init(args.config)

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

    init_distributed()

    main(model_args,
        data_args,
        training_args,
        inference_args,
        idm_args)
