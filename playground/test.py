import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from IDM.policy import InverseActionPolicy


device = "cuda"
model: InverseActionPolicy = InverseActionPolicy().to(device)

# (N, H, W, C)
shape = (1, 128,128,128,3)
agent_input = {"img": torch.rand(shape).to(device)}
hidden_state = model.initial_state(1)

inp = {
    "first": torch.zeros((agent_input["img"].shape[0], 1)).to(device),
    "state_in": hidden_state,
}

out, state_out = model.forward(agent_input, **inp)
(pi_logits, x, y) = out
print("done")