# Derived from HF Qwen3 implementation
# see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py

from typing import Callable, Optional, Union

import torch
from torch import nn
import einops
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ContextManagers
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.siglip import SiglipVisionModel
from cut_cross_entropy import linear_cross_entropy
from transformers.models.qwen3 import Qwen3Model
from transformers import AutoConfig, AutoProcessor


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def get_autocast_flags(self):
        return {
            "device_type": str(self.decoder[0].weight.device),
            "dtype": self.decoder[0].weight.dtype,
        }

    def forward(self, x):
        with torch.autocast(**self.get_autocast_flags()):
            out = self.decoder(x)
        return out

class LMAgent(GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()

        self.config = config    
        # Never initialize backbones here
        with ContextManagers(PreTrainedModel.get_init_context(False, False)):
            self.text_model = Qwen3Model(config["text_config"])
            self.vision_tower = SiglipVisionModel(config["vision_config"])

        self.hidden_dim: int = self.config.text_config.hidden_size
        self.vocab_size: int = self.config.text_config.vocab_size
        self.vision_hidden_dim: int = self.vision_tower.config.hidden_size
        self.mlp = MLP(self.vision_hidden_dim, self.hidden_dim)


    def get_processor(self):
        return None

    def cast_precision(self):
        self.text_model.to(dtype=torch.bfloat16)
        self.text_model.lm_head.to(dtype=torch.bfloat16)
        self.mlp.to(dtype=torch.bfloat16)
        self.vision_tower.to(dtype=torch.float32)

    def train_component(self, vision_tower=False, mlp=False, llm=False):
        for param in self.parameters():
            param.requires_grad = False

        param_groups = []

        if vision_tower:
            vt_params = list(
                set(self.vision_tower.vision_model.parameters()).difference(
                    set(self.vision_tower.vision_model.head.parameters())
                )
            )

            param_groups.append(vt_params)
            for param in vt_params:
                param.requires_grad = True

        if mlp:
            param_groups.append(self.mlp.parameters())
            for param in self.mlp.parameters():
                param.requires_grad = True

        if llm:
            param_groups.append(self.text_model.parameters())
            for param in self.text_model.parameters():
                param.requires_grad = True
        return param_groups

    def get_input_embeddings(self):
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.text_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```
        """

        B, T, C, H, W = pixel_values.shape

        pixel_values = einops.rearrange(pixel_values, "b s c h w -> (b s) c h w")

        vision_tokens = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        vision_tokens = einops.rearrange(
            vision_tokens, "(b s) c h -> b s c h", b=B, s=T
        )
        vision_tokens = self.mlp(vision_tokens)

        vision_tokens = einops.rearrange(vision_tokens, "b s t h -> b (s t) h")

        # (B,S,D) TODO add method to make this less ugly
        inputs_embeds = self.text_model.model.embed_tokens(input_ids)

        # replace the placeholder tokens with vision tokens
        V = vision_tokens.shape[1]
        inputs_embeds[:, :V, :] = torch.where(
            pixel_mask.unsqueeze(dim=-1) == 1, vision_tokens, inputs_embeds[:, :V, :]
        )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        loss = None
        logits = None

        if labels is not None:
            classifier = self.text_model.lm_head.weight
            loss = linear_cross_entropy(
                hidden_states, classifier, labels, shift=1, impl="cce_kahan_full"
            )
        else:  # assume infernece
            logits = self.text_model.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def init_vision_prcoessor(vision: str = None):
    processor = AutoProcessor(vision)
    return processor

def init_lm_agent(lm: str = None, vision: str = None)  -> LMAgent:
    from util.misc import local_model_map

    # TODO change this to not be hard coded
    lm_config_path = local_model_map(lm)
    vision_config_path = local_model_map(vision)
    
    config = {
        "text_config": AutoConfig.from_pretrained(lm_config_path),
        "vision_config": AutoConfig.from_pretrained(vision_config_path)
    }
    model = LMAgent(config)

    # init pretraiend weights
    model.text_model = Qwen3Model.from_pretrained(lm_config_path)
    model.vision_tower = SiglipVisionModel.from_pretrained(vision_config_path)
    return model