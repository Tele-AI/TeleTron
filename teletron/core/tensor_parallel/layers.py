import torch
import torch.nn as nn
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.parallel_state import get_tensor_model_parallel_world_size


from .mappings import tele_rmsnorm_cuisine, divide

__all__ = [
    "TeleColumnParallelLinear",
    "TeleRowParallelLinear",
    "TeleParallelRMSNorm",
]

class TeleColumnParallelLinear(ColumnParallelLinear):
    def forward(self, x):
        output, bias = super().forward(x)
        return output + bias if bias is not None else output

class TeleRowParallelLinear(RowParallelLinear):
    def forward(self, x):
        output, bias = super().forward(x)
        return output + bias if bias is not None else output
    
class TeleParallelRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        world_size = get_tensor_model_parallel_world_size()
        self.dim_per_partition = divide(dim, world_size)
        self.weight = nn.Parameter(torch.ones(self.dim_per_partition))
        
    def forward(self, x):
        return tele_rmsnorm_cuisine(x, self.weight, self.eps)

# Baseline

from .mappings import gather_from_tensor_model_parallel_region, scatter_to_tensor_model_parallel_region

class TeleParallelRMSNormBase(nn.Module):
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

    def forward(self, x_local):
        x = gather_from_tensor_model_parallel_region(x_local)
        dtype = x.dtype
        x_float = x.float()
        normalized_x = self.norm(x_float)
        normalized_x = normalized_x.to(dtype)
        output_global = normalized_x * self.weight
        output = scatter_to_tensor_model_parallel_region(output_global)
        return output
        
        