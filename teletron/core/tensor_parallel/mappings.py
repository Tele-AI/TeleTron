import torch
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""

    def ensure_divisibility(numerator, denominator):
        """Ensure that numerator is divisible by the denominator."""
        assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)

    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def _reduce_mean(input_):
    """All-reduce with mean operation on the input tensor across model parallel group."""
    
    # Bypass the function if we are using only 1 GPU.
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    
    # Create Copy
    output = input_.clone()
    
    # All reduce
    torch.distributed.all_reduce(output, group=get_tensor_model_parallel_group())
    
    # Get mean
    output /= world_size 
    
    return output

def _split_along_last_dim(input_):
    
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    
    # Calculate last dim
    last_dim = input_.dim() - 1
    last_dim_size = divide(input_.size()[last_dim], world_size)
    
    tensor_list = torch.split(input_, last_dim_size, dim=-1)
    rank = get_tensor_model_parallel_rank()
    output = tensor_list[rank].contiguous()
    
    return output


def _gather_along_last_dim(input_):
    """Gather tensors and concatenate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())
    
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output
    

class _TensorParallelMeanOperation(torch.autograd.Function):
    
    @staticmethod
    def symbolic(graph, input_):
        return _reduce_mean(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce_mean(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        world_size = get_tensor_model_parallel_world_size()
        grad_output /= world_size
        return grad_output
    
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)

class _GatherFromModelParallelRegion(torch.autograd.Function):
    
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)

class _TeleParallelRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        world_size = get_tensor_model_parallel_world_size()
        ctx.H = x.size(-1) * world_size
        
        orig_dtype = x.dtype
        x = x.float()
        weight = weight.float()
        
        x_squared = x.pow(2)
        mean_x_squared_local = x_squared.mean(dim=-1, keepdim=True)
        
        mean_x_squared = mean_x_squared_local.clone()
        torch.distributed.all_reduce(mean_x_squared, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
        mean_x_squared /= world_size
        
        rms = torch.sqrt(mean_x_squared + eps)
        rms_factor = torch.rsqrt(mean_x_squared + eps)
        normalized_x = x * rms_factor
        
        ctx.save_for_backward(x, weight, rms_factor, rms)
        ctx.orig_dtype = orig_dtype
        
        return (normalized_x * weight).to(orig_dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rms_factor, rms = ctx.saved_tensors
        eps = ctx.eps
        H = ctx.H
        orig_dtype = ctx.orig_dtype
        
        grad_output = grad_output.float()
        
        normalized_x = x * rms_factor
        grad_weight = (grad_output * normalized_x).sum(dim=(0, 1))
        
        w_grad_output = grad_output * weight
        partial_sum = (w_grad_output * normalized_x).sum(dim=-1, keepdim=True)
        
        global_sum = partial_sum.clone()
        torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        sum_part = global_sum / H
        grad_x = (w_grad_output - normalized_x * sum_part) * rms_factor
        
        return grad_x.to(orig_dtype), grad_weight.to(orig_dtype), None

# -----------------
# Helper functions.
# -----------------

def reduce_mean(input_):
    return _TensorParallelMeanOperation.apply(input_)

def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)

def tele_rmsnorm_cuisine(x, weight, eps):
    return _TeleParallelRMSNormFunction.apply(x, weight, eps)