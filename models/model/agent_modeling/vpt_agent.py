import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from emulator.keys import NUM_ACTION_CLASSES
from models.model.IDM.policy import PokePolicy
from models.util.dist import compute_accuracy

# memory_size == timesteps so maxlen=0 (causal attn within window only, no cross-batch KV state)
VPT_POLICY_KWARGS = {
    'attention_heads': 16,
    'attention_mask_style': 'clipped_causal',
    'attention_memory_size': 64,
    'hidsize': 1024,
    'img_shape': [128, 128, 3],
    'impala_kwargs': {'post_pool_groups': 1},
    'impala_width': 8,
    'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
    'n_recurrence_layers': 4,
    'only_img_input': True,
    'pointwise_ratio': 4,
    'pointwise_use_activation': False,
    'recurrence_is_residual': True,
    'recurrence_type': 'transformer',
    'single_output': True,
    'timesteps': 64,
    'use_pointwise_layer': True,
    'use_pre_lstm_ln': False,
}


class VPTAgent(nn.Module):
    """VPT-style policy: ImpalaCNN visual encoder + causal transformer."""

    def __init__(self, policy_kwargs=None):
        super().__init__()
        kwargs = policy_kwargs or VPT_POLICY_KWARGS
        self.net = PokePolicy(**kwargs)
        self.out_head = nn.Linear(kwargs['hidsize'], NUM_ACTION_CLASSES, bias=False)
        self.idm_labelling_fn = None

    def forward(
        self,
        pixel_values=None,   # (B, T, C, H, W) uint8 128x128
        labels=None,         # (B, S, C, H, W) uint8 128x128 for IDM labelling
        ground_labels=None,
        **kwargs,
    ):
        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device

        if labels is not None:
            input_ids = self.idm_labelling_fn(labels).detach()

        # PokePolicy expects (B, T, H, W, C) float
        frames = einops.rearrange(pixel_values, "b t c h w -> b t h w c").float()
        first = torch.zeros((B, 1), device=device)
        state_in = self.net.initial_state(B)  # empty KV state (maxlen=0)

        latent, _ = self.net({"img": frames}, state_in=state_in, context={"first": first})
        logits = self.out_head(latent)  # (B, T, num_actions)

        if labels is not None:
            loss = F.cross_entropy(
                einops.rearrange(logits, "b t c -> (b t) c"),
                einops.rearrange(input_ids, "b t -> (b t)")
            )
            with torch.no_grad():
                acc = compute_accuracy(logits, input_ids)
            return {"loss": loss, **acc}

        out = {"logits": logits}
        if ground_labels is not None:
            out |= compute_accuracy(logits, ground_labels.to(device=device), prefix="ground_")
        return out


def init_vpt_agent() -> VPTAgent:
    return VPTAgent()
