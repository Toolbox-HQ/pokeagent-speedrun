import torch
import torch.nn as nn
import transformers
from models.util.repro import repro_init
from models.model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from emulator.keys import CLASS_TO_KEY
from safetensors.torch import load_file
import math
from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments
from models.train.train_idm import IDMArguments
from pprint import pprint
    
class PokeAgentActionConditioned:
    def __init__(self, model_path: str, device: str, temperature = 0.01, actions_per_second = 60, model_fps = 2, context_len = 64, sampling_strategy="default"):

        pprint({
            "model_path": model_path,
            "device": device,
            "temperature": temperature,
            "actions_per_second": actions_per_second,
            "model_fps": model_fps,
            "context_len": context_len,
            "sampling_strategy": sampling_strategy,
        })
         
        self.sampling_strategy = sampling_strategy
        self.context_len = context_len
        self.actions_per_second = actions_per_second
        self.model_fps = model_fps
        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = actions_per_second // self.model_fps
        
        self.device = torch.device(device)
        self.temperature = temperature

        self.model: nn.Module = init_lm_agent(lm="Qwen/Qwen3-1.7B", vision="google/siglip-base-patch16-224", use_cache=True)
        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.processor = init_vision_prcoessor("google/siglip-base-patch16-224", use_cache=True)


        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8) 
        self.input_ids = torch.zeros(1, self.buffersize, dtype=torch.long) 
        self.idx = 0

        print(f"buffersize: {self.buffersize}")

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        if self.idx == self.buffersize - 1:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame

        images = self.agent_frames[self.idx % self.stride::self.stride]

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)

        input_ids_dev = self.input_ids.to(self.device)[:, self.idx % self.stride::self.stride]

        output = self.model(input_ids=input_ids_dev, pixel_values=pixel_values)
        logits = output["logits"][0, math.floor(self.idx / self.stride)]                     # (num_classes,)

        if self.sampling_strategy == "default":
            cls = torch.argmax(logits, dim=-1)
        elif self.sampling_strategy == "temperature":
            probs = torch.softmax(logits / self.temperature, dim=-1)       # temperature sampling
            cls = torch.multinomial(probs, num_samples=1).squeeze(-1)      # sample an index

        if self.idx == 0: # Go right
            cls = torch.tensor(7, dtype=torch.long, device=probs.device)

        if self.idx == self.buffersize - 1:
            self.input_ids = torch.cat((self.input_ids[:, 1:], cls.view(1, 1).to('cpu')), dim=1)
        else:
            self.input_ids[0, self.idx] = cls.to('cpu')

        if self.idx < self.buffersize - 1:
            self.idx += 1

        return CLASS_TO_KEY[int(cls.item())]

class PokeagentStateOnly:
    def __init__(self, model_path: str, device: str, temperature = 0.01, actions_per_second = 60, model_fps = 2, context_len = 64, sampling_strategy="default"):

        pprint({
            "model_path": model_path,
            "device": device,
            "temperature": temperature,
            "actions_per_second": actions_per_second,
            "model_fps": model_fps,
            "context_len": context_len,
            "sampling_strategy": sampling_strategy,
        })
         
        self.sampling_strategy = sampling_strategy
        self.context_len = context_len
        self.actions_per_second = actions_per_second
        self.model_fps = model_fps
        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = actions_per_second // self.model_fps
        
        self.device = torch.device(device)
        self.temperature = temperature

        self.model: nn.Module = init_lm_agent(arch="state_only", lm="Qwen/Qwen3-1.7B", vision="google/siglip-base-patch16-224", use_cache=True)
        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.processor = init_vision_prcoessor("google/siglip-base-patch16-224", use_cache=True)


        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8)  
        self.idx = 0

        print(f"buffersize: {self.buffersize}")

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        if self.idx == self.buffersize - 1:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame

        # start = self.idx % self.stride
        # indices = list(range(start, len(self.agent_frames), self.stride))
        
        # print(indices)
        images = self.agent_frames[self.idx % self.stride::self.stride]
        # print(f"length:{len(indices)}")

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)

        output = self.model(pixel_values=pixel_values)
        logits = output["logits"][0, math.floor(self.idx / self.stride)]                     # (num_classes,)
        # print(f"sampledidx: {indices[math.floor(self.idx / self.stride)]}")
        # print(f"sampledlogit: {math.floor(self.idx / self.stride)}")

        if self.mode == "default":
            cls = torch.argmax(logits, dim=-1)
        elif self.sampling_strategy == "temperature":
            probs = torch.softmax(logits / self.temperature, dim=-1)       # temperature sampling
            cls = torch.multinomial(probs, num_samples=1).squeeze(-1)      # sample an index
        #print(probs)

        if self.idx == 0: # Go right
            cls = torch.tensor(7, dtype=torch.long, device=probs.device)

        if self.idx < self.buffersize - 1:
            self.idx += 1

        return CLASS_TO_KEY[int(cls.item())]

from models.train.train_agent import setup_training, train, create_dataset, init_model
from models.train.train_idm import train_idm

class OnlinePokeagentStateOnly:
    def __init__(self,
                model_args: ModelArguments,
                training_args: TrainingArguments,
                data_args: DataArguments,
                inference_args: InferenceArguments,
                idm_args: IDMArguments
                ):
        
        self.model_args: ModelArguments = model_args
        self.training_args: TrainingArguments = training_args
        self.data_args: DataArguments = data_args
        self.inference_args: InferenceArguments = inference_args
        self.idm_args: IDMArguments = idm_args

        assert inference_args.model_checkpoint is None, "Use model_args.load_path"
         
        self.sampling_strategy = inference_args.sampling_strategy
        self.context_len = inference_args.context_length
        self.actions_per_second = inference_args.actions_per_seconds
        self.model_fps = inference_args.agent_fps
        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = self.actions_per_second // self.model_fps
        self.temperature = inference_args.temperature

        self.model, self.idm, self.processor, self.device = init_model(self.model_args, self.training_args)
        self.model.load_state_dict(load_file(model_args.load_path))
        self.model.to(self.device).eval()
        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8)  
        self.idx = 0
        print("[AGENT] Initialized agent")
    
    def train_agent(self, intervals: str):
        train_ds, eval_ds = create_dataset(intervals, self.processor)
        self.model.train()
        train(self.model, self.training_args, train_ds=train_ds, eval_ds=eval_ds)
        self.model.eval()

    def train_idm(self, data_dir: str):
        self.idm.train()
        train_idm(self.idm, self.idm_args, data_dir)
        self.idm.eval()

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        if self.idx == self.buffersize - 1:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame
        
        images = self.agent_frames[self.idx % self.stride::self.stride]

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)

        output = self.model(pixel_values=pixel_values)
        logits = output["logits"][0, math.floor(self.idx / self.stride)]                     # (num_classes,)

        if self.mode == "default":
            cls = torch.argmax(logits, dim=-1)
        elif self.sampling_strategy == "temperature":
            probs = torch.softmax(logits / self.temperature, dim=-1)       # temperature sampling
            cls = torch.multinomial(probs, num_samples=1).squeeze(-1)      # sample an index

        if self.idx == 0: # Go right
            cls = torch.tensor(7, dtype=torch.long, device=probs.device)

        if self.idx < self.buffersize - 1:
            self.idx += 1

        return CLASS_TO_KEY[int(cls.item())]

class OnlinePokeagent:
    def __init__(self,
                model_args: ModelArguments,
                training_args: TrainingArguments,
                data_args: DataArguments,
                inference_args: InferenceArguments,
                idm_args: IDMArguments
                ):
    
        self.model_args: ModelArguments = model_args
        self.training_args: TrainingArguments = training_args
        self.data_args: DataArguments = data_args
        self.inference_args: InferenceArguments = inference_args
        self.idm_args: IDMArguments = idm_args

        assert inference_args.model_checkpoint is None, "Use model_args.load_path"

        self.context_len = self.inference_args.context_length
        self.actions_per_second = self.inference_args.actions_per_seconds
        self.model_fps = self.inference_args.agent_fps

        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = 60 // self.model_fps

        
        self.temperature = self.inference_args.temperature
        self.model, self.idm, self.processor, self.device = init_model(self.model_args, self.training_args)
        self.model.load_state_dict(load_file(model_args.load_path))
        self.model.to(self.device).eval()
        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8)  
        self.idx = 0
        print("[AGENT] Initialized agent")

    def train_agent(self, intervals: str):
        train_ds, eval_ds = create_dataset(intervals, self.processor)
        self.model.train()
        train(self.model, self.training_args, train_ds=train_ds, eval_ds=eval_ds)
        self.model.eval()

    def train_idm(self, data_dir: str):
        self.idm.train()
        train_idm(self.idm, self.idm_args, data_dir)
        self.idm.eval()

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        if self.idx == self.buffersize - 1:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame

        images = self.agent_frames[self.idx % self.stride::self.stride]
        S = images.shape[0]
        if S < self.context_len:
            pad = torch.zeros((self.context_len - S, *images.shape[1:]), dtype=images.dtype, device=images.device,)
            images = torch.cat((images, pad), dim=0)

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)

        output = self.model(pixel_values=pixel_values)
        logits = output["logits"][0, math.floor(self.idx / self.stride)]                         # (num_classes,)
        probs = torch.softmax(logits / self.temperature, dim=-1)       # temperature sampling
        cls = torch.multinomial(probs, num_samples=1).squeeze(-1)      # sample an index
        #print(probs)

        if self.idx < self.buffersize - 1:
            self.idx += 1

        return CLASS_TO_KEY[int(cls.item())]
    
if __name__ == "__main__":

    
    from argparse import ArgumentParser

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

    pokeagent = OnlinePokeagent(model_args=model_args,
                                training_args=training_args,
                                data_args=data_args,
                                inference_args=inference_args,
                                idm_args=idm_args)
    
    #pokeagent.train_agent(".cache/pokeagent/agent_data/early_game.json")
    pokeagent.train_idm("pokeagent/test_filter")