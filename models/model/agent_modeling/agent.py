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
from torch.nn import Module
from models.policy import NUM_ACTION_CLASSES

# TODO this should refactored to be a util
from models.train.train_idm import compute_accuracy


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

class LMAgent(Module, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tconfig = config["text_config"]
        self.vconfig = config["vision_config"].vision_config
        self.num_actions = config["num_actions"]
        self.idm_labelling_fn = None

        with ContextManagers(PreTrainedModel.get_init_context(False, False)):
            self.text_model = Qwen3Model(self.tconfig)
            self.vision_tower = SiglipVisionModel(self.vconfig)

        self.hidden_dim: int = self.tconfig.hidden_size
        self.vocab_size: int = self.tconfig.vocab_size
        self.vision_hidden_dim: int = self.vision_tower.config.hidden_size
        self.mlp = MLP(self.vision_hidden_dim, self.hidden_dim)
        self.action_embedding = nn.Embedding(self.num_actions, self.hidden_dim)
        self.output_actions = nn.Linear(self.hidden_dim, self.num_actions, bias=False)
        self.finish_init()

    def get_processor(self):
        return None

    def finish_init(self):
        delattr(self.text_model, "embed_tokens")
        self.cast_mixed_precision()

    def cast_mixed_precision(self):
        self.text_model.to(dtype=torch.bfloat16)
        self.action_embedding.to(dtype=torch.bfloat16)
        self.output_actions.to(dtype=torch.bfloat16)
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
        input_ids: Optional[torch.LongTensor] = None, # important
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None, # important
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # important
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        # TODO All these vars should be re-named
        if labels is not None:
            input_ids = self.idm_labelling_fn(labels).detach()

        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        pixel_values = einops.rearrange(pixel_values, "b s c h w -> (b s) c h w")

        vision_tokens = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        vision_tokens = einops.rearrange(vision_tokens, "(b s) c h -> b s c h", b=B, s=T)
        vision_tokens = self.mlp(vision_tokens)
        B, T, S, H = vision_tokens.shape
        
        action_tokens = self.action_embedding(input_ids)

        hiddens = torch.zeros((B, S+1, H), dtype=torch.int32, device=device)
        
        hiddens[:, -1, :] = 1
        hiddens[:,-2, :] = -1
        hiddens = hiddens.repeat(1,T,1)
        
        state_mask = hiddens < 1 # all visual tokens
        action_mask = hiddens == 1 # all action tokens
        last_token_mask = hiddens == -1 # last visual token, used for prediction

        hiddens = hiddens.to(torch.bfloat16)
        hiddens = hiddens.masked_scatter(state_mask, vision_tokens.to(torch.bfloat16))
        hiddens = hiddens.masked_scatter(action_mask, action_tokens)


        if not attention_mask:
            B, S, H = hiddens.shape
            attention_mask = torch.ones((B, S), dtype=torch.int32, device=device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=hiddens,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        action_hiddens = hidden_states[last_token_mask].view(B, T, H)
        out = {}

        if input_ids is not None:
            out["loss"] = linear_cross_entropy(action_hiddens.contiguous(), self.output_actions.weight, input_ids)

        # eval and inference
        with torch.no_grad():
            logits = self.output_actions(action_hiddens)
            
            if input_ids is not None:
                out |= compute_accuracy(logits, input_ids)
            else:
                out = {"logits": self.output_actions(action_hiddens)}

        return out

class LMStateAgent(Module, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tconfig = config["text_config"]
        self.vconfig = config["vision_config"].vision_config
        self.num_actions = config["num_actions"]
        self.idm_labelling_fn = None

        with ContextManagers(PreTrainedModel.get_init_context(False, False)):
            self.text_model = Qwen3Model(self.tconfig)
            self.vision_tower = SiglipVisionModel(self.vconfig)

        self.hidden_dim: int = self.tconfig.hidden_size
        self.vocab_size: int = self.tconfig.vocab_size
        self.vision_hidden_dim: int = self.vision_tower.config.hidden_size
        self.mlp = MLP(self.vision_hidden_dim, self.hidden_dim)
        self.output_actions = nn.Linear(self.hidden_dim, self.num_actions, bias=False)
        self.finish_init()

    def get_processor(self):
        return None

    def finish_init(self):
        delattr(self.text_model, "embed_tokens")
        self.cast_mixed_precision()

    def cast_mixed_precision(self):
        self.text_model.to(dtype=torch.bfloat16)
        self.output_actions.to(dtype=torch.bfloat16)
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
        input_ids: Optional[torch.LongTensor] = None, # important
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None, # important
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # important
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        # TODO All these vars should be re-named
        if labels is not None:
            input_ids = self.idm_labelling_fn(labels).detach()

        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        pixel_values = einops.rearrange(pixel_values, "b s c h w -> (b s) c h w")

        vision_tokens = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        vision_tokens = einops.rearrange(vision_tokens, "(b s) c h -> b s c h", b=B, s=T)
        vision_tokens = self.mlp(vision_tokens)
        B, T, S, H = vision_tokens.shape
        hiddens: torch.Tensor = einops.rearrange(vision_tokens, "b t s h -> b (t s) h", b=B, t=T, s=S, h=H).to(torch.bfloat16)


        if not attention_mask:
            B, X, H = hiddens.shape
            attention_mask = torch.ones((B, X), dtype=torch.int32, device=device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=hiddens,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        action_hiddens = hidden_states[:,::S,:]
        out = {}

        if input_ids is not None:
            out["loss"] = linear_cross_entropy(action_hiddens.contiguous(), self.output_actions.weight, input_ids)

        # eval and inference
        with torch.no_grad():
            logits = self.output_actions(action_hiddens)
            
            if input_ids is not None:
                out |= compute_accuracy(logits, input_ids)
            else:
                out = {"logits": self.output_actions(action_hiddens)}

        return out

def init_vision_prcoessor(vision: str = None, use_cache: bool = True):
    from models.util.misc import local_model_map
    if use_cache:
        return AutoProcessor.from_pretrained(local_model_map(vision))
    else:
        return AutoProcessor.from_pretrained(vision)

def init_lm_agent(arch: str = "default", lm: str = None, vision: str = None, use_cache: bool = True)  -> nn.Module:
    from models.util.misc import local_model_map

    if use_cache:
        lm = local_model_map(lm)
        vision = local_model_map(vision) 

    print(f"use_cache set to {use_cache}")
    print(f"loading vision encoder from {vision}")
    print(f"loading llm from {lm}")

    config = {
        "num_actions": NUM_ACTION_CLASSES,
        "text_config": AutoConfig.from_pretrained(lm),
        "vision_config": AutoConfig.from_pretrained(vision)
    }
    if arch == "default":
        model = LMAgent(config)
    elif arch == "state_only":
        model = LMStateAgent(config)
    else:
        raise Exception("NotImplementedError")

    model.text_model = Qwen3Model.from_pretrained(lm)
    model.vision_tower = SiglipVisionModel.from_pretrained(vision)
    model.finish_init()

    return model