# Copyright (c) 2025 TeleAI-infra and Nvidia Megatron-LM Team. All rights reserved.


from typing import Optional, List, Any

import torch
import torch.distributed as dist
from torch import Tensor
from megatron.core import mpu


ALL_TO_ALL_BUFFER = None

def set_global_all_to_all_buffer(buffer_numel):
    global ALL_TO_ALL_BUFFER
    cp_size = mpu.get_context_parallel_world_size()
    # define all_to_all input buffer
    buffer_numel = buffer_numel // cp_size
    ALL_TO_ALL_BUFFER = [torch.zeros(buffer_numel, device=torch.cuda.current_device(), dtype=torch.bfloat16) for _ in range(cp_size)]

def get_all_to_all_buffer(loca_input):
    global ALL_TO_ALL_BUFFER
    # resize ALL_TO_ALL_BUFFER
    cp_size = mpu.get_context_parallel_world_size()
    input_numel = loca_input.numel() // cp_size
    if input_numel > ALL_TO_ALL_BUFFER[0].numel():
        ALL_TO_ALL_BUFFER = None
        ALL_TO_ALL_BUFFER = [torch.zeros(input_numel, device=torch.cuda.current_device(), dtype=torch.bfloat16) for _ in range(cp_size)]
    all_to_all_buffer = [t[:input_numel] for t in ALL_TO_ALL_BUFFER]
    return all_to_all_buffer


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        use_buffer: bool = False,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        ctx.use_buffer = use_buffer
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, use_buffer, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.use_buffer, ctx.group, False),
            None,
            None,
            None,
            None,
        )

def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    use_buffer: bool,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = mpu.get_context_parallel_group() if group is None else group
    cp_size = dist.get_world_size(group)
    input_shape = list(local_input.shape)
    input_shape[scatter_dim] = input_shape[scatter_dim] // cp_size

    if use_buffer:
        all_to_all_buffer = get_all_to_all_buffer(local_input)
        all_to_all_input = list(torch.tensor_split(local_input, cp_size, scatter_dim))
        input_list = [a.copy_(b.reshape(-1)) for a,b in zip(all_to_all_buffer, all_to_all_input)]
    else:
        input_list = [t.contiguous() for t in torch.tensor_split(local_input, cp_size, scatter_dim)]
    
    # 创建输出buffer
    output_list = [torch.empty(input_shape, device=torch.cuda.current_device(), dtype=torch.bfloat16) for _ in range(cp_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()

def split_forward_gather_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    split_sizes: Optional[List[int]] = None,
    grad_scale: str = "down"

) -> torch.Tensor:
    """
    Splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.
    This function supports both aligned and unaligned data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to split the tensor.
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "down".

    Returns:
        torch.Tensor: The resulting tensor after splitting and keeping only the corresponding chunk.
    """
    
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[List[int]] = None,
    grad_scale: str = "up"
) -> torch.Tensor:
    """
    Gathers the input tensor from all processes in the model parallel region and concatenates them along the specified
    dimension. During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.
    This function handles both aligned and unaligned data during the gather and scatter operations.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to concatenate the gathered tensors.
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

    Returns:
        torch.Tensor: The resulting tensor after gathering and concatenating.
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)

class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Custom autograd function that gathers the input tensor from all processes in the model parallel region and
    concatenates them.
    During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.

    """

    @staticmethod
    def symbolic(graph, input_, process_group, dim, gather_sizes):
        """
        Define the symbolic representation of the custom operation.
        """
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        """
        Forward pass: Gathers tensors from all processes in the specified process group and concatenates them along the specified dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            dim (int): The dimension along which to concatenate the gathered tensors.
            gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

        Returns:
            torch.Tensor: The resulting tensor after gathering and concatenating.
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, ctx.gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Distribute the gradients to the input tensors and scales them according to the gradient scaling mode.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            torch.Tensor: The gradient of the input with respect to the loss.
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Custom autograd function that splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.

    """
    @staticmethod
    def symbolic(graph, input_, process_group, dim, split_sizes):
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.split_sizes = split_sizes
        return _split(input_, process_group, dim, ctx.split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None



def _split(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Splits a tensor across the specified dimension and returns the part corresponding to the current rank,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be split.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to split the tensor. Defaults to -1 (last dimension).
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes, with the remainder
            distributed to the first few processes. Defaults to None.

    Returns:
        torch.Tensor: The part of the tensor corresponding to the current rank in the process group.
    """
    # Ensure split_sizes is a list if provided
    assert split_sizes is None or isinstance(split_sizes, list)

    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    # Calculate split sizes if not provided
    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size

        # Calculate the size for each process
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)

    # Get the part corresponding to the current rank
    rank = dist.get_rank(pg)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: Optional[List[int]] = None):
    """
    Gathers tensors from all processes in the process group and concatenates them along the specified dimension,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be gathered.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to concatenate the gathered tensors. Defaults to -1 (last dimension).
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.

    Returns:
        torch.Tensor: The concatenated tensor after gathering from all processes in the process group.
    """
    # Ensure gather_sizes is a list if provided
    assert gather_sizes is None or isinstance(gather_sizes, list)

    # Skip if only one rank is involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()
    tensor_shape_base = list(input_.size())
    assert input_.device.type == "cuda"
    # Prepare the output list with appropriate shapes
    if gather_sizes:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
            torch.distributed.all_gather(tensor_list, input_, group=pg)
            output = torch.cat(tensor_list, dim=dim)
    else:
        tensor_shape_base[dim] = tensor_shape_base[dim] * world_size
        output = torch.empty(tensor_shape_base, dtype=input_.dtype, device=input_.device)
        torch.distributed._all_gather_base(output, input_, group=pg)

    return output.contiguous()