import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core import mpu
from yunchang.comm.all_to_all import SeqAllToAll4D
from .mappings import split_forward_gather_backward, gather_forward_split_backward, SeqAllToAll
from .layers import GateWithGradReduce, ModulateWithCPGradReduce
from teletron.utils import get_args

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
    
class ContextParallelMixin:

    @staticmethod
    def cp_grad_reduce(grad):
        with torch.no_grad():
            # cp_size = mpu.get_context_parallel_world_size()
            # dim_size = list(grad.size())
            # dim_size[0] = dim_size[0] * cp_size
            # grad_list = torch.empty(dim_size, dtype=grad.dtype, device=torch.cuda.current_device())
            # torch.distributed._all_gather_base(grad_list, grad.contiguous(), group=mpu.get_context_parallel_group())
            # grad_list = grad_list.view(cp_size, -1, *grad_list.shape[1:])
            # reduced_grad = grad_list.sum(dim=0)
            # allreduce
            reduced_grad = grad.contiguous()
            torch.distributed.all_reduce(reduced_grad, group=mpu.get_context_parallel_group())
        
        return reduced_grad

    def enable_context_parallel(self, attn_module: nn.Module):
        attn_module.forward = self.forward_attn
    
    def split_input(self, x, dim):
        # assume x is not parallel
        cp_group = mpu.get_context_parallel_group()

        x = self.pad_for_context_parallel(x, dim)
        x = split_forward_gather_backward(x, cp_group, dim=dim, grad_scale="none")
        return x
    
    def gather_output(self, output, dim):
        # assume output is parallel
        cp_group = mpu.get_context_parallel_group()
        output = gather_forward_split_backward(output, cp_group, dim=dim, grad_scale="none")
        output = self.remove_pad_for_context_parallel(output, dim)
        return output 

    @staticmethod
    def pad_for_context_parallel(tensor, dim):
        cp_size = mpu.get_context_parallel_world_size()
        ContextParallelMixin.origin_length = tensor.shape[dim]
        ContextParallelMixin.padded_length = math.ceil(ContextParallelMixin.origin_length / cp_size) * cp_size
        pad_size = int(ContextParallelMixin.padded_length - ContextParallelMixin.origin_length)

        if pad_size <= 0:
            return tensor  # No padding needed

        # Create pad tuple: (dim_n_before, dim_n_after, ..., dim_0_before, dim_0_after)
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * dim + 1)] = pad_size  # pad after the dimension
        return torch.nn.functional.pad(tensor, pad) 
    
    @staticmethod
    def remove_pad_for_context_parallel(tensor, dim):
        # remove pad must be called after pad
        return tensor.narrow(dim, 0, ContextParallelMixin.origin_length)

    @staticmethod
    def remove_pad_with_encoder_for_context_parallel(tensor, encoder_length, dim):
        total_length = tensor.size(dim)
        
        split_point = total_length - encoder_length
        first_raw = tensor.narrow(dim, 0, split_point)
        first = first_raw.narrow(dim, 0, ContextParallelMixin.origin_length)

        second = tensor.narrow(dim, split_point, encoder_length)

        result = torch.cat([first, second], dim=dim)
        return result

    def forward_attn(self, q, k, v):
        cp_group = mpu.get_context_parallel_group()
        args = get_args()
        num_heads = args.num_attention_heads // mpu.get_tensor_model_parallel_world_size()
        
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)

        # qkv: b s/CP n d
        if mpu.get_context_parallel_world_size() > 1:
            q = SeqAllToAll.apply(cp_group, q, 2, 1, True)
            k = SeqAllToAll.apply(cp_group, k, 2, 1, True)
            v = SeqAllToAll.apply(cp_group, v, 2, 1, True)

            # qkv: b s n/CP d
            q,k,v = map(
                lambda x: self.remove_pad_for_context_parallel(x, 1),
                [q,k,v]
            )

        if FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_func(q, k, v)[0]
            x = x.transpose(1, 2).contiguous()
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            x = F.scaled_dot_product_attention(q, k, v)
        if mpu.get_context_parallel_world_size() > 1:
            x = self.pad_for_context_parallel(x, 2)
            x = SeqAllToAll.apply(
                cp_group, x, 2, 1, True
            )  # b img_seq sub_n d

            # x: b n s/CP d
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h

        return x

    def gate_with_cp_grad_reduce(self, x, gate, residual):
        return GateWithGradReduce.apply(x, gate, residual)
    
    def modulate_with_cp_grad_reduce(self, x, shift, scale):
        return ModulateWithCPGradReduce.apply(x, shift, scale)
