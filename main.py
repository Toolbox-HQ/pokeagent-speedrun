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

def checkpoint(output_dir: str, step: int, agent, emulator):
    import os
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    agent_model = agent.model
    agent_idm = agent.idm
    
    if dist.get_rank() == 0:
        print(f"[GPU {dist.get_rank()} LOOP] save checkpoint at step {step} to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(agent_idm.state_dict(), os.path.join(save_path, f"idm_model.pt"))
        save_file(agent_model.state_dict(), os.path.join(save_path, f"agent.safetensors"))
        emulator.save_state(os.path.join(save_path, f"game.state"))

def run_online_agent(model_args, data_args, training_args, inference_args, idm_args, output_dir, run_uuid: str): 
    from models.inference.agent_inference import OnlinePokeagentStateOnly, OnlinePokeagentStateActionConditioned
    from emulator.emulator_connection import EmulatorConnection
    from tqdm import tqdm
    import numpy as np
    import torch
    import uuid
    from concurrent.futures import ThreadPoolExecutor, wait
    from models.inference.find_matching_videos import get_videos
    import json
    import torch.distributed as dist
    from models.util.misc import finalize_wandb
    import os

    rank = dist.get_rank()

    with open(inference_args.save_state, 'rb') as f:
        curr_state = f.read()
    
    if inference_args.inference_architecture == "state_only":
        agent = OnlinePokeagentStateOnly(model_args, training_args, data_args, inference_args, idm_args)
    elif inference_args.inference_architecture == "state_action_conditioned":
        agent = OnlinePokeagentStateActionConditioned(model_args, training_args, data_args, inference_args, idm_args)
    else:
        raise Exception(f"{inference_args.inference_architecture} is not supported")
    
    video_path = f'{output_dir}/runs/output_gpu{rank}'
    bootstrap_count = 0

    agent_data_path = f'{output_dir}/agent_data'
    idm_data_path = f'{output_dir}/idm_data'
    query_path_template = f'{output_dir}/query_video/query_gpu{rank}_bootstrap'
    idm_data_path_template = f'{idm_data_path}/idm_gpu{rank}_bootstrap'
    dino_embedding_path = '.cache/pokeagent/db_embeddings'
    agent_path_template = f'{agent_data_path}/videos_gpu{rank}_bootstrap'
    checkpoint_path =  f'{output_dir}/checkpoints/online_debug'
    query_path = query_path_template + str(bootstrap_count)

    conn = EmulatorConnection(inference_args.rom_path)
    conn.load_state(curr_state)
    conn.create_video_writer(query_path)
    conn.start_video_writer(query_path)
    conn.create_video_writer(video_path)
    conn.start_video_writer(video_path)

    futures = []

    with ThreadPoolExecutor(max_workers=100) as executor:
        for step in tqdm(range(inference_args.agent_steps), desc=f"GPU {rank} Agent Exploration"):
            if step != 0 and step % inference_args.bootstrap_interval == 0:
                wait(futures)
                futures.clear()
                
                conn.release_video_writer(query_path)

                dist.barrier()
                print(f"[GPU {rank} LOOP] Begin IDM training")

                agent.train_idm(idm_data_path) # train idm on cumulative idm data
                finalize_wandb(tags = [run_uuid, "idm", f"bootstrap_{bootstrap_count}"])
                print(f"[GPU {rank} LOOP] IDM training completed")

                video_intervals = get_videos(f"{query_path}.mp4",
                                                dino_embedding_path,
                                                inference_args.match_length,
                                                inference_args.retrieved_videos,
                                                inference_args.max_vid_len
                                                )
                print(f"[GPU {rank} LOOP] Finished Retrieval")
                
                video_path = agent_path_template + f"{bootstrap_count}.json"
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                with open(video_path, "w") as f:
                    json.dump(video_intervals, f)
                print(f"[GPU {rank} LOOP] Saved intervals")


                dist.barrier()
                print(f"[GPU {rank} LOOP] Begin agent training")

                agent.train_agent(agent_data_path) # train agent on cumulative agent data
                finalize_wandb(tags = [run_uuid, "agent", f"bootstrap_{bootstrap_count}"])
                print(f"[GPU {rank} LOOP] Agent training completed")

                bootstrap_count += 1
                query_path = query_path_template + str(bootstrap_count)
               
                checkpoint(checkpoint_path, step, agent, conn)

                conn.create_video_writer(query_path)
                conn.start_video_writer(query_path)


            if step % inference_args.idm_data_sample_interval == 0:
                id = str(uuid.uuid4())
                new_conn = EmulatorConnection(inference_args.rom_path)
                new_conn.load_state(conn.get_state())
                futures.append(executor.submit(run_random_agent, new_conn, inference_args.idm_data_sample_steps, idm_data_path_template + f"{bootstrap_count}/{id}"))

            tensor = torch.from_numpy(np.array(conn.get_current_frame())).permute(2, 0, 1) # CHW, uint8
            key = agent.infer_action(tensor)
            conn.run_frames(7)
            conn.set_key(key)
            conn.run_frames(8)

    conn.release_video_writer(query_path)
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


def main(model_args, data_args, training_args, inference_args, idm_args, output_dir, uuid):
   
    if inference_args.online:
        run_online_agent(model_args, data_args, training_args, inference_args, idm_args, output_dir, uuid)
    else:
        run_agent(*inference_args)

if __name__ == "__main__":
    import torch.distributed as dist
    import os
    from argparse import ArgumentParser
    import transformers
    from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments
    from models.train.train_idm import IDMArguments
    from models.util.repro import repro_init
    from models.util.dist import init_distributed, clean_dist_and_exit, get_shared_uuid

    init_distributed()

    uuid: str = get_shared_uuid()
    rank: int = dist.get_rank()

    if rank == 0:
        print(f"[RUN UUID]: {uuid}")

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    checkpoint_path = repro_init(args.config)

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
        idm_args,
        os.path.join(training_args.output_dir, uuid),
        uuid
        )
    
    clean_dist_and_exit()
