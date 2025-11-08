import torch
from models.model.agent_modeling.agent import init_lm_agent, init_vision_processor
from models.policy.policy import CLASS_TO_KEY
from safetensors.torch import load_file

class Pokeagent:
    def __init__(self, device: str):
        self.device = torch.device(device)

        self.model = init_lm_agent(lm="Qwen/Qwen3-1.7B", vision="google/siglip-base-patch16-224")
        state_dict = load_file(".cache/model.safetensors")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.processor = init_vision_processor("google/siglip-base-patch16-224")

        self.agent_frames = torch.zeros(64, 3, 160, 240)            
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
        cls = torch.argmax(output.logits[0, self.idx], dim=-1)

        if self.idx == 63:
            self.input_ids = torch.cat((self.input_ids[:, 1:], cls.view(1, 1).to('cpu')), dim=1)
        else:
            self.input_ids[0, self.idx] = cls.to('cpu')

        if self.idx < 63:
            self.idx += 1

        return CLASS_TO_KEY[int(cls.item())]