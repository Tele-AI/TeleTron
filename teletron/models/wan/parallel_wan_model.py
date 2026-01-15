# Copyright (c) 2025 TeleAI-infra Team and The HuggingFace Team. All rights reserved.

from typing import Optional
import torch
import torch.nn as nn

from teletron.core.context_parallel import ContextParallelMixin
from teletron.core.transformer import TransformerGeneralMixin
from .wan_model import WanModel, DiTBlock, sinusoidal_embedding_1d
from teletron.utils import set_config

class ContextParallelWanDitBlock(ContextParallelMixin, DiTBlock):
    def __init__(self, *args, **kwargs):
        DiTBlock.__init__(self, *args, **kwargs)
        # from ContextParallelMixin
        self.enable_context_parallel(self.self_attn.attn)

    def forward(self, x, context, t_mod, freqs):
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        modulation = modulation + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        normed_x1 = self.norm1(x)
        modulated_x1 = self.modulate_with_cp_grad_reduce(normed_x1, shift_msa, scale_msa)
        attn_output = self.self_attn(modulated_x1, freqs)
        gated_x1 = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)

        normed_x3 = self.norm3(gated_x1)
        cross_attn_output = self.cross_attn(normed_x3, context)
        x = gated_x1 + cross_attn_output

        normed_x2 = self.norm2(x)
        modulated_x2 = self.modulate_with_cp_grad_reduce(normed_x2, shift_mlp, scale_mlp)
        ffn_output = self.ffn(modulated_x2)
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, ffn_output)

        return x


class ParallelWanModel(ContextParallelMixin, TransformerGeneralMixin, WanModel):
    def __init__(self, config):
        dit_model_config = set_config().get('model_config', None).get('dit', None).get('config', None)
        WanModel.__init__(self, **dit_model_config)
        self.config = config
        
        self.blocks = nn.ModuleList([
            ContextParallelWanDitBlock(self.has_image_input, self.dim, self.num_heads, self.ffn_dim, self.eps)
            for _ in range(self.num_layers)
        ])

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

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cn_images=None, 
        **kwargs,
    ):
        # Do whatever necessary before forward transformer blocks
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if y is not None:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        
        if self.has_image_input:
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)  # (b, c_x + c_y, f, h, w)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # Split input sequence and rope (with methods from CPMixin), and forward CP transformer blocks
        x = self.split_input(x, dim=1)
        freqs = self.split_input(freqs, dim=0)
        x = self.blocks(x, context, t_mod, freqs)
        x = self.gather_output(x, dim=1)

        # Now x is in full shape, just do regular forward
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def sharded_state_dict(self):
        return self.state_dict()
