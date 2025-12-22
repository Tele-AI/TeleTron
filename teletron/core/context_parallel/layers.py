import torch 
import torch.nn as nn
from megatron.core import mpu 
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

class ContextParallelGateModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, gate, residual):
        return GateWithGradReduce.apply(x, gate, residual)


def gate_with_cp_grad_reduce(x, gate, residual):
    return GateWithGradReduce.apply(x, gate, residual)


class GateWithGradReduce(torch.autograd.Function ):
    @staticmethod
    def forward(ctx, x, gate, residual):
        ctx.save_for_backward(gate, residual)
        return x + gate * residual
    
    @staticmethod
    def backward(ctx, x_grad):
        gate, residual = ctx.saved_tensors
        r_grad = x_grad * gate 
        gate_grad = torch.sum((x_grad * residual), dim=1, keepdim=True)
        torch.distributed.all_reduce(gate_grad, group=mpu.get_context_parallel_group())
        return x_grad, gate_grad, r_grad


class ModulateWithCPGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale):
        ctx.save_for_backward(x, scale)
        return (x * (1 + scale) + shift)
    
    @staticmethod 
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        x_grad = grad_output * (1 + scale) 
        scale_grad = torch.sum((grad_output * x), dim=1, keepdim=True)
        torch.distributed.all_reduce(scale_grad, group=mpu.get_context_parallel_group())
        shift_grad = torch.sum(grad_output, dim=1, keepdim=True)
        torch.distributed.all_reduce(shift_grad, group=mpu.get_context_parallel_group())
        return x_grad, shift_grad, scale_grad


def modulate_with_cp_grad_reduce(x, shift, scale):
    return ModulateWithCPGradReduce.apply(x, shift, scale)

