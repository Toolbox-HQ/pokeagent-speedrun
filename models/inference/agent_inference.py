import torch
import torch.nn as nn
import torch.distributed as dist
from models.model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from emulator.keys import CLASS_TO_KEY
from safetensors.torch import load_file
import math
from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments, IDMArguments
from pprint import pprint
from models.train.train_agent import create_dataset, init_model, train_with_rollback, AgentObjectiveManager
from models.train.train_idm import train_idm_best_checkpoint
from transformers import AutoImageProcessor, AutoModel
from models.util.misc import local_model_map
import pickle
    
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

        #print(f"buffersize: {self.buffersize}")

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

        return CLASS_TO_KEY[cls.item()]


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

        return CLASS_TO_KEY[cls.item()]

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
        self.actions_per_second = inference_args.actions_per_second
        self.model_fps = inference_args.agent_fps
        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = self.actions_per_second // self.model_fps
        self.temperature = inference_args.temperature

        self.model, self.idm, self.processor, self.device = init_model(self.model_args, self.training_args)
        self.model.load_state_dict(load_file(model_args.load_path))
        self.model.to(self.device).eval()
        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8, device=self.device)  
        self.idx = 0
        print("[AGENT] Initialized agent")
    
    def train_agent(self, intervals: str, bootstrap: int, query_embeds: list, video_frames: list):
        train_ds, eval_ds, objective_manager = create_dataset(intervals, self.processor, bootstrap, split = self.inference_args.train_eval_split, query_embeds=query_embeds, video_frames=video_frames, data_args=self.data_args)
        self.model.train()
        train_with_rollback(self.model, self.training_args, train_ds=train_ds, eval_ds=eval_ds)
        self.model.eval()
        self.objective_manager = objective_manager

    def train_idm(self, data_dir: str):
        self.idm.train()
        train_idm_best_checkpoint(self.idm,
                                  self.idm_args,
                                  data_dir,
                                  split = self.inference_args.train_eval_split)
        self.idm.eval()

    def broadcast_agent_state(self, src=1):
        if dist.is_initialized():
            dist.broadcast(self.agent_frames, src=src)
            input_ids_device = self.input_ids.to(self.device)
            dist.broadcast(input_ids_device, src=src)
            self.input_ids = input_ids_device.to('cpu')
            print(F"[AGENT] Broadcasted agent state from GPU {src}")

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)
        
        frame = frame.to(self.device)

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

        return CLASS_TO_KEY[cls.item()]

class OnlinePokeagentStateActionConditionedObjective:
    def __init__(self,
                model_args: ModelArguments,
                training_args: TrainingArguments,
                data_args: DataArguments,
                inference_args: InferenceArguments,
                idm_args: IDMArguments,
                ):
        
        self.model_args: ModelArguments = model_args
        self.training_args: TrainingArguments = training_args
        self.data_args: DataArguments = data_args
        self.inference_args: InferenceArguments = inference_args
        self.idm_args: IDMArguments = idm_args

        assert inference_args.model_checkpoint is None, "Use model_args.load_path"
         
        self.sampling_strategy = inference_args.sampling_strategy
        self.context_len = inference_args.context_length
        self.actions_per_second = inference_args.actions_per_second
        self.model_fps = inference_args.agent_fps
        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = self.actions_per_second // self.model_fps
        self.temperature = inference_args.temperature

        self.num_objectives = 10

        self.model, self.idm, self.processor, self.device = init_model(self.model_args, self.training_args)
        print(f"****************{model_args.load_path}")
        self.model.load_state_dict(load_file(model_args.load_path))
        self.model.to(self.device).eval()
        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8, device=self.device)
        self.input_ids = torch.zeros(1, self.buffersize, dtype=torch.long)
        self.idx = 0
        self.total_steps = 0

        with open(model_args.objective_load_path, 'rb') as f:
            self.objective_manager = pickle.load(f)

        dino_id = local_model_map("facebook/dinov2-base")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(dino_id).eval().to(self.device)
        print("[AGENT] Initialized agent")
     
    def train_agent(self, intervals: str, bootstrap: int, query_embeds: list, video_frames: list):
        train_ds, eval_ds, objective_manager = create_dataset(intervals, self.processor, bootstrap, split = self.inference_args.train_eval_split, query_embeds=query_embeds, video_frames=video_frames, data_args=self.data_args)
        self.model.train()
        train_with_rollback(self.model, self.training_args, train_ds=train_ds, eval_ds=eval_ds)
        self.model.eval()
        self.objective_manager = objective_manager

    def train_idm(self, data_dir: str):
        self.idm.train()
        train_idm_best_checkpoint(self.idm, self.idm_args, data_dir, split = self.inference_args.train_eval_split)
        self.idm.eval()

    def broadcast_agent_state(self, src=1):
        if dist.is_initialized():
            dist.broadcast(self.agent_frames, src=src)
            input_ids_device = self.input_ids.to(self.device)
            dist.broadcast(input_ids_device, src=src)
            self.input_ids = input_ids_device.to('cpu')

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        frame = frame.to(self.device)

        if self.idx == self.buffersize - 1:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame

        images = self.agent_frames[self.idx % self.stride::self.stride]

        self.total_steps += 1
        if self.total_steps % self.context_len == 0:
            dino_inputs = self.dino_processor(images=images.cpu(), return_tensors="pt")
            dino_inputs = {k: v.to(self.device) for k, v in dino_inputs.items()}
            dino_outputs = self.dino_model(**dino_inputs)
            embeds = dino_outputs.last_hidden_state[:, 0]  # CLS token, (S, D)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            self.objective_manager.mine_and_add_objectives(
                [e.cpu().numpy() for e in embeds],
                images.cpu()
            )

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)

        objectives_list = self.objective_manager.retrieve_last_n_objectives(self.num_objectives)
        n_pad = self.num_objectives - len(objectives_list)
        if objectives_list:
            obj_pixels = self.processor(images=[o.cpu() for o in objectives_list], return_tensors="pt")["pixel_values"]
            if n_pad > 0:
                obj_pixels = torch.cat([obj_pixels, torch.zeros(n_pad, *obj_pixels.shape[1:])], dim=0)
        else:
            obj_pixels = torch.zeros(self.num_objectives, *pixel_values.shape[2:])
        objectives = obj_pixels.to(self.device).unsqueeze(0)  # (1, num_objectives, C, H, W)

        input_ids_dev = self.input_ids.to(self.device)[:, self.idx % self.stride::self.stride]

        output = self.model(input_ids=input_ids_dev, pixel_values=pixel_values, objectives=objectives)
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

        return CLASS_TO_KEY[cls.item()]
    

class OnlinePokeagentStateActionConditioned:
        def __init__(self,
                model_args: ModelArguments,
                training_args: TrainingArguments,
                data_args: DataArguments,
                inference_args: InferenceArguments,
                idm_args: IDMArguments,
                objective_manager: AgentObjectiveManager
                ):
        
            self.model_args: ModelArguments = model_args
            self.training_args: TrainingArguments = training_args
            self.data_args: DataArguments = data_args
            self.inference_args: InferenceArguments = inference_args
            self.idm_args: IDMArguments = idm_args
            self.objective_manager = objective_manager

            assert inference_args.model_checkpoint is None, "Use model_args.load_path"
            
            self.sampling_strategy = inference_args.sampling_strategy
            self.context_len = inference_args.context_length
            self.actions_per_second = inference_args.actions_per_second
            self.model_fps = inference_args.agent_fps
            self.buffersize = self.context_len // self.model_fps * self.actions_per_second
            self.stride = self.actions_per_second // self.model_fps
            self.temperature = inference_args.temperature

            self.model, self.idm, self.processor, self.device = init_model(self.model_args, self.training_args)
            self.model.load_state_dict(load_file(model_args.load_path))
            self.model.to(self.device).eval()
            self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8, device=self.device) 
            self.input_ids = torch.zeros(1, self.buffersize, dtype=torch.long) 
            self.idx = 0
            print("[AGENT] Initialized agent")
     
    def train_agent(self, data_dir: str, bootstrap: int):
        train_ds, eval_ds, objective_manager = create_dataset(data_dir, self.processor, bootstrap, split = self.inference_args.train_eval_split, data_args=self.data_args)
        self.model.train()
        train_with_rollback(self.model, self.training_args, train_ds=train_ds, eval_ds=eval_ds)
        self.model.eval()
        self.objective_manager = objective_manager

    def train_idm(self, data_dir: str):
        self.idm.train()
        train_idm_best_checkpoint(self.idm, self.idm_args, data_dir, split = self.inference_args.train_eval_split)
        self.idm.eval()

    def broadcast_agent_state(self, src=1):
        if dist.is_initialized():
            dist.broadcast(self.agent_frames, src=src)
            input_ids_device = self.input_ids.to(self.device)
            dist.broadcast(input_ids_device, src=src)
            self.input_ids = input_ids_device.to('cpu')

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        frame = frame.to(self.device)

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

        return CLASS_TO_KEY[cls.item()]