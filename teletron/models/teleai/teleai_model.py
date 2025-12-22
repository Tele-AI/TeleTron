import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

T5_CONTEXT_TOKEN_NUMBER = 512

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, device="cuda"):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_RoPE(x, freqs, num_heads):
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_float64 = x.to(torch.float64)
    x_reshaped = x_float64.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    x_rotated = x_complex * freqs
    x_real = torch.view_as_real(x_rotated)
    x_flattened = x_real.reshape(batch_size, seq_len, num_heads * head_dim)
    x_out = x_flattened.to(x.dtype)
    return x_out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        x_squared = x.pow(2)
        mean_x_squared = x_squared.mean(dim=-1, keepdim=True)
        mean_x_squared_eps = mean_x_squared + self.eps
        rms_norm_factor = torch.rsqrt(mean_x_squared_eps)
        normalized_x = x * rms_norm_factor
        return normalized_x

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        normalized_x = self.norm(x_float)
        normalized_x = normalized_x.to(dtype)
        output = normalized_x * self.weight
        return output


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_query = RMSNorm(dim, eps=eps)
        self.norm_key = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_query(self.query(x))
        k = self.norm_key(self.key(x))
        v = self.value(x)
        q = apply_RoPE(q, freqs, self.num_heads)
        k = apply_RoPE(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.out_proj(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_query = RMSNorm(dim, eps=eps)
        self.norm_key = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.img_key = nn.Linear(dim, dim)
            self.img_value = nn.Linear(dim, dim)
            self.norm_image_key = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.attn2 = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            image_context_length = y.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            img = y[:, :image_context_length]  # Image tokens
            ctx = y[:, image_context_length:]  # Context tokens (e.g., text)
        else:
            ctx = y  # Only context tokens if no image input

        q = self.query(x)
        q = self.norm_query(q)

        k = self.key(ctx)
        k = self.norm_key(k)

        v = self.value(ctx)

        x_attn = self.attn(q, k, v)

        if self.has_image_input:
            k_img = self.img_key(img)
            k_img = self.norm_image_key(k_img)
            v_img = self.img_value(img)
            y_attn = self.attn2(q, k_img, v_img)
            x_attn = x_attn + y_attn

        output = self.out_proj(x_attn)
        return output


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.gate2 = GateModule()

    def forward(self, x, context, t_mod, freqs):
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        normalized_x_msa = self.norm1(x)
        modulated_x_msa = modulate(normalized_x_msa, shift_msa, scale_msa)
        self_attention_output = self.self_attn(modulated_x_msa, freqs)
        gated_self_attention = self.gate(x, gate_msa, self_attention_output)

        normalized_x_cross = self.norm3(gated_self_attention)
        cross_attention_output = self.cross_attn(normalized_x_cross, context)
        x_with_cross_attention = gated_self_attention + cross_attention_output

        normalized_x_mlp = self.norm2(x_with_cross_attention)
        modulated_x_mlp = modulate(normalized_x_mlp, shift_mlp, scale_mlp)
        mlp_output = self.ffn(modulated_x_mlp)
        output = self.gate2(x_with_cross_attention, gate_mlp, mlp_output)

        return output


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift, scale = modulation.chunk(2, dim=1)
        normalized_x = self.norm(x)
        scaled_x = normalized_x * (1 + scale)
        shifted_x = scaled_x + shift
        projected_x = self.head(shifted_x)
        return projected_x


class TeleaiModel(torch.nn.Module):
    def __init__(
        self, 
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.has_image_input = has_image_input
        self.has_image_pos_emb = has_image_pos_emb

        self.patch_emb = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_emb = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.dim, self.dim)
        )
        self.time_emb = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(self.has_image_input, self.dim,
                     self.num_heads, self.ffn_dim, self.eps)
            for _ in range(self.num_layers)
        ])
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        if self.has_image_input:
            # clip_feature_dim = 1280
            self.img_emb = MLP(
                1280, self.dim, has_pos_emb=self.has_image_pos_emb)

    def patchify(self, x: torch.Tensor):
        x = self.patch_emb(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        cn_images=None,
        **kwargs,
    ):
        # Compute the time embedding using a sinusoidal embedding followed by a feedforward network
        time_embedding = sinusoidal_embedding_1d(self.freq_dim, timestep)
        t = self.time_emb(time_embedding)
        projected_time = self.time_proj(t)
        modified_t = projected_time.unflatten(1, (6, self.dim))

        # Project the context (text) embedding to the model dimension
        context_emb = self.text_emb(context)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        if self.has_image_input:            
            clip_emb = self.img_emb(clip_feature)
            context_emb = torch.cat([clip_emb, context_emb], dim=1)

        # If conditional images are provided, concatenate them to the input
        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)

        # Patchify the input and get the grid size
        x, grid_size = self.patchify(x)
        f, h, w = grid_size

        # Compute 3D rotary positional embeddings for the patches
        head_dim = self.dim // self.num_heads
        freq_f = precompute_freqs_cis(head_dim - 2 * (head_dim // 3), f).view(f, 1, 1, -1).expand(f, h, w, -1)
        freq_h = precompute_freqs_cis(head_dim // 3, h).view(1, h, 1, -1).expand(f, h, w, -1)
        freq_w = precompute_freqs_cis(head_dim // 3, w).view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([freq_f, freq_h, freq_w], dim=-1)
        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context_emb, modified_t, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context_emb, modified_t, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context_emb, modified_t, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
    
