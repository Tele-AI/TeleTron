import torch
from typing import Optional, Tuple
from .modules.longcat_video_dit import LongCatVideoTransformer3DModel

class LongcatVideoModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: Optional[dict] = None,
        cp_split_hw: Optional[Tuple[int, int]] = None,
        text_tokens_zero_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dit = LongCatVideoTransformer3DModel(
            in_channels=in_dim,
            out_channels=out_dim,
            hidden_size=dim,
            depth=num_layers,
            num_heads=num_heads,
            caption_channels=text_dim,
            mlp_ratio=mlp_ratio,
            adaln_tembed_dim=adaln_tembed_dim,
            frequency_embedding_size=freq_dim,
            patch_size=patch_size,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
            text_tokens_zero_pad=text_tokens_zero_pad,
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        img_emb_y: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        return_kv: bool = False,
        kv_cache_dict: dict = {},
        skip_crs_attn: bool = False,
        offload_kv_cache: bool = False,
        **kwargs,
    ):
        self.dit.gradient_checkpointing = use_gradient_checkpointing

        if img_emb_y is not None:
            assert img_emb_y.dim() == 5 and latents.dim() == 5
            assert img_emb_y.shape[:1] == latents.shape[:1]
            latents = torch.cat([img_emb_y, latents], dim=2)
            num_cond_latents = img_emb_y.shape[2]
        else:
            num_cond_latents = 0

        return self.dit.forward(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
            num_cond_latents=num_cond_latents,
            return_kv=return_kv,
            kv_cache_dict=kv_cache_dict,
            skip_crs_attn=skip_crs_attn,
            offload_kv_cache=offload_kv_cache,
        )
