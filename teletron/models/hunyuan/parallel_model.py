from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import mpu
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention_processor import Attention
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from teletron.core.context_parallel import ContextParallelMixin
from teletron.core.transformer import TransformerGeneralMixin
from teletron.core.context_parallel.mappings import split_forward_gather_backward,\
    gather_forward_split_backward, SeqAllToAll
from .model import HunyuanVideoTransformer3DModel
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

class HunyuanVideoDoubleAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_length = encoder_hidden_states.shape[1]
        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # 4. Encoder condition QKV projection and normalization
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)
        

        # Ulysses Context Parallel AlltoAll of qkv
        cp_group = mpu.get_context_parallel_group()

        query = SeqAllToAll.apply(cp_group, query, 1, 2)
        key = SeqAllToAll.apply(cp_group, key,  1, 2)
        value = SeqAllToAll.apply(cp_group, value,  1, 2)
        query, key ,value = map(
                lambda x: ContextParallelMixin.remove_pad_for_context_parallel(x, dim=2),
                [query, key, value]
            )
        
        added_query = split_forward_gather_backward(encoder_query, cp_group, dim=1)
        added_key = split_forward_gather_backward(encoder_key, cp_group, dim=1)
        added_value= split_forward_gather_backward(encoder_value, cp_group, dim=1)
        del encoder_query, encoder_key, encoder_value

        query = torch.cat([query, added_query], dim=2)
        key = torch.cat([key, added_key], dim=2)
        value = torch.cat([value, added_value], dim=2)

        ### 
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=1)
        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : -encoder_length, :, :],
            hidden_states[:, -encoder_length :, :, :],
        )
        
        # 6. Output projection
        hidden_states = ContextParallelMixin.pad_for_context_parallel(hidden_states, 1)
        hidden_states = SeqAllToAll.apply(cp_group, hidden_states, 1, 2)
        encoder_hidden_states = gather_forward_split_backward(encoder_hidden_states, cp_group, 2)
        
        hidden_states = hidden_states.flatten(2, 3).contiguous()
        encoder_hidden_states = encoder_hidden_states.flatten(2, 3).contiguous()
        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        torch.cuda.empty_cache()
        return hidden_states, encoder_hidden_states
    
class HunyuanVideoSingleAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoSingleAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        img_query = query[:, :, : -encoder_length]
        txt_query = query[:, :, -encoder_length :]
        img_key = key[:, :, : -encoder_length]
        txt_key = key[:, :, -encoder_length :]
        img_value = value[:, :, : -encoder_length]
        txt_value = value[:, :, -encoder_length :]
        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            img_query = apply_rotary_emb(img_query, image_rotary_emb)
            img_key = apply_rotary_emb(img_key, image_rotary_emb)
        
        # Ulysses Context Parallel AlltoAll of qkv
        cp_group = mpu.get_context_parallel_group()

        query = SeqAllToAll.apply(cp_group, img_query,  1, 2)
        key = SeqAllToAll.apply(cp_group, img_key, 1, 2)
        value = SeqAllToAll.apply(cp_group, img_value, 1, 2)

        added_query = split_forward_gather_backward(txt_query, cp_group, dim=1)
        added_key = split_forward_gather_backward(txt_key, cp_group, dim=1)
        added_value= split_forward_gather_backward(txt_value, cp_group, dim=1)
        del img_query, img_key, img_value, txt_query, txt_key, txt_value

        query = torch.cat([query, added_query], dim=2)
        key = torch.cat([key, added_key], dim=2)
        value = torch.cat([value, added_value], dim=2)

        del added_query, added_key, added_value
        query, key ,value = map(
                lambda x: ContextParallelMixin.remove_pad_with_encoder_for_context_parallel(x, encoder_length, dim=2),
                [query, key, value]
            )
        ### 
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=1)
        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : -encoder_length, :, :],
            hidden_states[:, -encoder_length :, :, :],
        )

        # 6. Output projection
        hidden_states = ContextParallelMixin.pad_for_context_parallel(hidden_states, 1)
        hidden_states = SeqAllToAll4D.apply(cp_group, hidden_states, 1, 2)
        encoder_hidden_states = gather_forward_split_backward(encoder_hidden_states, cp_group, 2)
        
        hidden_states = hidden_states.flatten(2, 3).contiguous()
        encoder_hidden_states = encoder_hidden_states.flatten(2, 3).contiguous()
        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        torch.cuda.empty_cache()
        return hidden_states, encoder_hidden_states

class ParallelHunyuanVideoModel(ContextParallelMixin, TransformerGeneralMixin, HunyuanVideoTransformer3DModel):
    def __init__(self, config):
        super().__init__(config)
        # from TransformerGeneralMixin
        self.enable_activation_checkpointing(self.transformer_blocks)
        self.enable_activation_checkpointing(self.single_transformer_blocks)

        # from ContextParallelMixin
        for i in range(len(self.transformer_blocks)):
            self.transformer_blocks[i].attn.processor = HunyuanVideoDoubleAttnProcessor2_0()
            self.transformer_blocks[i].norm1.modulate = self.modulate_with_cp_grad_reduce
            self.transformer_blocks[i].gate = self.gate_with_cp_grad_reduce
        for i in range(len(self.single_transformer_blocks)):
            self.single_transformer_blocks[i].attn.processor = HunyuanVideoSingleAttnProcessor2_0()
            self.single_transformer_blocks[i].norm.modulate = self.modulate_with_cp_grad_reduce
            self.single_transformer_blocks[i].gate = self.gate_with_cp_grad_reduce

        self.register_cp_grad_reduce_hook()

    
    def register_cp_grad_reduce_hook(self):
        
        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook     
        self.wgrad_not_to_reduce =[
                f"single_transformer_blocks.{i}.norm.linear.weight" for i in range(self.num_single_layers)
            ] + [
                f"single_transformer_blocks.{i}.norm.linear.bias" for i in range(self.num_single_layers)
            ] + [
                f"transformer_blocks.{i}.norm1.linear.weight" for i in range(self.num_layers)
            ] + [
                f"transformer_blocks.{i}.norm1.linear.bias"  for i in range(self.num_layers)
            ]
        for name, param in self.named_parameters():
            if name.startswith("time_text_embed") or\
                name.startswith("x_embedder") or \
                    name.startswith("norm_out") or \
                        name.startswith("proj_out")  or "modulation" in name:
                continue 
            if name not in self.wgrad_not_to_reduce:
                param.register_hook(ContextParallelMixin.cp_grad_reduce)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.patch_size, self.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        freqs_cos, freqs_sin = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        # 3. Attention mask preparation
        if encoder_attention_mask is None:
            attention_mask = None
        else:
            latent_sequence_length = hidden_states.shape[1]
            condition_sequence_length = encoder_hidden_states.shape[1]
            sequence_length = latent_sequence_length + condition_sequence_length
            attention_mask = torch.zeros(
                batch_size,
                sequence_length,
                sequence_length,
                device=hidden_states.device,
                dtype=torch.bool,
            )  # [B, N, N]

            effective_condition_sequence_length = encoder_attention_mask.sum(
                dim=1, dtype=torch.int
            )  # [B,]
            effective_sequence_length = (
                latent_sequence_length + effective_condition_sequence_length
            )

            for i in range(batch_size):
                attention_mask[
                    i, : effective_sequence_length[i], : effective_sequence_length[i]
                ] = True

        # 4. Transformer blocks
        hidden_states = self.split_input(hidden_states, dim=1)
        freqs_cos = self.split_input(freqs_cos, dim=0)
        freqs_sin = self.split_input(freqs_sin, dim=0)

        image_rotary_emb = (freqs_cos, freqs_sin)
        hidden_states, encoder_hidden_states = self.transformer_blocks(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            image_rotary_emb,
        )
        hidden_states, encoder_hidden_states = self.single_transformer_blocks(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            image_rotary_emb,
        )
        hidden_states = self.gather_output(hidden_states, dim=1)

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)