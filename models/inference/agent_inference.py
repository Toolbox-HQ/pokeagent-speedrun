import torch
from models.model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from emulator.keys import CLASS_TO_KEY
from safetensors.torch import load_file

class Pokeagent:
    def __init__(self, device: str, temperature = 0.01):
        self.device = torch.device(device)
        self.temperature = temperature

        self.model = init_lm_agent(lm="Qwen/Qwen3-1.7B", vision="google/siglip-base-patch16-224", use_cache=False)
        state_dict = load_file(".cache/agent.safetensors")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.processor = init_vision_prcoessor("google/siglip-base-patch16-224", use_cache=False)
        self.model.training = False

        self.agent_frames = torch.zeros(64, 3, 160, 240, dtype=torch.uint8)            
        self.input_ids = torch.zeros(1, 64, dtype=torch.long)       
        self.idx = 0

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        if self.idx == 63:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame

        inputs = self.processor(images=self.agent_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)
        input_ids_dev = self.input_ids.to(self.device)

        output = self.model(input_ids=input_ids_dev, pixel_values=pixel_values)
        logits = output["logits"][0, self.idx]                         # (num_classes,)
        probs = torch.softmax(logits / self.temperature, dim=-1)            # temperature sampling
        cls = torch.multinomial(probs, num_samples=1).squeeze(-1)      # sample an index
        #print(probs)

        if self.idx == 0:
            cls = torch.tensor(7, dtype=torch.long, device=probs.device)

        if self.idx == 63:
            self.input_ids = torch.cat((self.input_ids[:, 1:], cls.view(1, 1).to('cpu')), dim=1)
        else:
            self.input_ids[0, self.idx] = cls.to('cpu')

        if self.idx < 63:
            self.idx += 1

        return CLASS_TO_KEY[int(cls.item())]