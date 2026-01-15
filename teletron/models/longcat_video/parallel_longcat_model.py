# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.amp as amp
from einops import rearrange
import numpy as np

from teletron.core.context_parallel import ContextParallelMixin
from teletron.core.transformer import TransformerGeneralMixin
from teletron.core.context_parallel.layers import modulate_with_cp_grad_reduce, gate_with_cp_grad_reduce
from .modules.longcat_video_dit import LongCatVideoTransformer3DModel, LongCatSingleStreamBlock
from .modules.blocks import modulate_fp32
from teletron.utils import set_config

class ContextParallelLongCatDitBlock(ContextParallelMixin, LongCatSingleStreamBlock):
    def __init__(self, *args, **kwargs):
        LongCatSingleStreamBlock.__init__(self, *args, **kwargs)
        # We do NOT enable context parallel on attn because LongCat's Attention module 
        # already handles CP splitting/gathering internally via bsa/ulysses.
        # self.enable_context_parallel(self.attn) 

    def forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None, skip_crs_attn=False):
        """
            x: [B, N, C]
            y: [1, N_valid_tokens, C]
            t: [B, T, C_t]
            y_seqlen: [B]; type of a list
            latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape # S != T*H*W in case of CP split on H*W.
        S = N // T

        # compute modulation params in fp32
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1) # [B, T, 1, C]

        # self attn with modulation
        # x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
        
        # Adapt for ContextParallelMixin's modulate_with_cp_grad_reduce which sums over dim=1
        # We reshape [B, T, S, C] -> [B*T, S, C] so dim=1 is S (the split dimension)
        x_reshaped = x.view(B * T, S, C)
        shift_msa_reshaped = shift_msa.reshape(B * T, 1, C)
        scale_msa_reshaped = scale_msa.reshape(B * T, 1, C)
        
        normed_x = self.mod_norm_attn(x_reshaped)
        x_m = modulate_with_cp_grad_reduce(normed_x, shift_msa_reshaped, scale_msa_reshaped)
        x_m = x_m.view(B, N, C).to(x_dtype)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache)
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv)
        
        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            # x = x + (gate_msa * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
            # Use gate_with_cp_grad_reduce
            # x + gate * residual. Here x is 'x', gate is 'gate_msa', residual is 'x_s'
            gate_msa_reshaped = gate_msa.reshape(B * T, 1, C)
            x_s_reshaped = x_s.view(B * T, S, C)
            x_residual_reshaped = x.view(B * T, S, C)
            
            x = gate_with_cp_grad_reduce(x_residual_reshaped, gate_msa_reshaped, x_s_reshaped)
            x = x.view(B, N, C)
            
        x = x.to(x_dtype)

        # cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)

        # ffn with modulation
        # x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        shift_mlp_reshaped = shift_mlp.reshape(B * T, 1, C)
        scale_mlp_reshaped = scale_mlp.reshape(B * T, 1, C)
        
        x_reshaped = x.view(B * T, S, C)
        normed_x = self.mod_norm_ffn(x_reshaped)
        x_m = modulate_with_cp_grad_reduce(normed_x, shift_mlp_reshaped, scale_mlp_reshaped)
        x_m = x_m.view(B, N, C).to(x_dtype) # Reshape back for FFN which expects [B, N, C] or similar
        
        # FeedForwardSwiGLU expects [..., C]
        x_s = self.ffn(x_m)
        
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            # x = x + (gate_mlp * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
            gate_mlp_reshaped = gate_mlp.reshape(B * T, 1, C)
            x_s_reshaped = x_s.view(B * T, S, C)
            x_residual_reshaped = x.view(B * T, S, C)
            
            x = gate_with_cp_grad_reduce(x_residual_reshaped, gate_mlp_reshaped, x_s_reshaped)
            x = x.view(B, N, C)
            
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class ParallelLongCatModel(ContextParallelMixin, TransformerGeneralMixin, LongCatVideoTransformer3DModel):
    def __init__(self, config):
        # Read TeleAI-style config and map to LongCat args; filter by constructor signature for robustness
        dit_cfg_root = set_config().get('model_config', None).get('dit', None)
        dit_model_config = dit_cfg_root.config
        dim = getattr(dit_model_config, 'dim', getattr(dit_model_config, 'hidden_size', 4096))
        num_layers = getattr(dit_model_config, 'num_layers', getattr(dit_model_config, 'depth', 48))
        num_heads = getattr(dit_model_config, 'num_heads', 32)
        text_dim = getattr(dit_model_config, 'text_dim', getattr(dit_model_config, 'caption_channels', 4096))
        in_dim = getattr(dit_model_config, 'in_dim', getattr(dit_model_config, 'in_channels', 16))
        out_dim = getattr(dit_model_config, 'out_dim', getattr(dit_model_config, 'out_channels', 16))
        ffn_dim = getattr(dit_model_config, 'ffn_dim', None)
        freq_dim = getattr(dit_model_config, 'freq_dim', getattr(dit_model_config, 'frequency_embedding_size', 256))
        adaln_tembed_dim = getattr(dit_model_config, 'adaln_tembed_dim', 512)
        patch_size = getattr(dit_model_config, 'patch_size', getattr(dit_model_config, 'patch_size', [1, 2, 2]))
        text_tokens_zero_pad = getattr(dit_model_config, 'text_tokens_zero_pad', False)
        has_image_input = getattr(dit_model_config, 'has_image_input', False)
        has_image_pos_emb = getattr(dit_model_config, 'has_image_pos_emb', False)
        enable_flashattn3 = getattr(dit_model_config, 'enable_flashattn3', False)
        enable_flashattn2 = getattr(dit_model_config, 'enable_flashattn2', False)
        enable_xformers = getattr(dit_model_config, 'enable_xformers', False)
        enable_bsa = getattr(dit_model_config, 'enable_bsa', False)
        bsa_params = getattr(dit_model_config, 'bsa_params', None)
        cp_split_hw = getattr(dit_model_config, 'cp_split_hw', None)
        # Derive mlp_ratio from ffn_dim if provided
        mlp_ratio = getattr(dit_model_config, 'mlp_ratio', None)
        if mlp_ratio is None:
            if ffn_dim is not None and dim > 0:
                mlp_ratio = max(1, int(round(ffn_dim / dim)))
            else:
                mlp_ratio = 4
        # Build argument dict and filter by LongCatVideoTransformer3DModel.__init__ signature
        import inspect
        init_sig = inspect.signature(LongCatVideoTransformer3DModel.__init__)
        allowed_params = set(init_sig.parameters.keys())
        arg_map = dict(
            in_channels=in_dim,
            out_channels=out_dim,
            hidden_size=dim,
            depth=num_layers,
            num_heads=num_heads,
            caption_channels=text_dim,
            mlp_ratio=mlp_ratio,
            adaln_tembed_dim=adaln_tembed_dim,
            frequency_embedding_size=freq_dim,
            patch_size=tuple(patch_size),
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
            text_tokens_zero_pad=text_tokens_zero_pad,
            has_image_input=has_image_input,
            has_image_pos_emb=has_image_pos_emb,
        )
        safe_args = {k: v for k, v in arg_map.items() if k in allowed_params}
        LongCatVideoTransformer3DModel.__init__(self, **safe_args)
        self.parallel_config = config
        
        # Replace blocks with ContextParallel-aware blocks
        # We need to extract arguments from the existing blocks to recreate them
        # Or just use the init params
        
        # Params from LongCatVideoTransformer3DModel init
        hidden_size = self.blocks[0].hidden_size
        num_heads = self.blocks[0].attn.num_heads
        # mlp_ratio = self.blocks[0].ffn.hidden_dim / hidden_size # inferred
        # We can also get these from dit_model_config if available
        
        # Better to re-instantiate using the same params as passed to __init__
        # assuming dit_model_config has all of them.
        
        self.blocks = nn.ModuleList(
            [
                ContextParallelLongCatDitBlock(
                    hidden_size=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw
                )
                for _ in range(num_layers)
            ]
        )

        # from TransformerGeneralMixin
        from teletron.utils import get_args
        args = get_args()
        if args.activation_offload:
            self.enable_activation_offload(self.blocks)
        else:
            self.enable_activation_checkpointing(self.blocks)

        # from ContextParallelMixin
        self.register_cp_grad_reduce_hook()

    def register_cp_grad_reduce_hook(self):
        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook
        for name, param in self.named_parameters():
            if name.startswith("patch_embedding") or \
                    name.startswith("time") or\
                        name.startswith("head") or \
                             "modulation" in name:
                continue 

            param.register_hook(self.cp_grad_reduce)
            
    @property
    def config(self):
        return self.parallel_config

    # We do NOT override forward() because LongCatVideoTransformer3DModel.forward 
    # already handles 2D CP splitting/gathering via context_parallel_util.split_cp_2d
    # which is compatible with the blocks we swapped in.

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def sharded_state_dict(self):
        return self.state_dict()
