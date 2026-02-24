def run_random_agent(inference_args, training_args):
    from models.inference.random_agent import RandomAgent
    from emulator.emulator_connection import EmulatorConnection
    from emulator.keys import KEY_LIST_FOR_IDM
    from tqdm import tqdm
    import os


    video_path = os.path.join(training_args.output_dir, "random_agent")
    os.makedirs(training_args.output_dir, exist_ok=True)

    conn = EmulatorConnection(inference_args.rom_path)

    
    if inference_args.save_state is not None:
        with open(inference_args.save_state, 'rb') as f:
            state = f.read()
        conn.load_state(state)

    conn.create_video_writer(video_path)
    conn.start_video_writer(video_path)

    agent = RandomAgent(30, 180, KEY_LIST_FOR_IDM)
    for _ in tqdm(range(inference_args.agent_steps)):
        key, num_frames = agent.infer()
        conn.set_key(key)
        conn.run_frames(num_frames)

    conn.release_video_writer(video_path)
    conn.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import transformers
    from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments
    from models.train.train_idm import IDMArguments

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    hf_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, InferenceArguments, IDMArguments)
    )
    model_args, data_args, training_args, inference_args, idm_args = hf_parser.parse_yaml_file(yaml_file=args.config)

    run_random_agent(inference_args, training_args)
