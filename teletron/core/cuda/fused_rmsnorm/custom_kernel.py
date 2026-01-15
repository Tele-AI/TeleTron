import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Function
from torch.nn.parameter import Parameter


import custom_rmsnorm

class RMSNormModelFunction(Function):
    """
    Implements custom RMSNorm operation as a PyTorch autograd Function.
    Uses fused CUDA kernels for accelerated computation. Currently optimized for cols=128 only.
    """
    
    @staticmethod
    def forward(ctx, x, weight, epsilon, cols):
        """
        Forward pass for RMSNorm operation.
        
        Args:
            ctx: Context object for saving tensors needed in backward pass
            x: Input tensor of shape
            weight: Learnable scaling parameter of shape [cols]
            epsilon: Small constant for numerical stability (1e-6 typical)
            cols: Feature dimension size (must be 128 in current implementation)
            
        Returns:
            output: Normalized tensor with same shape as input x

        """
        x = x.contiguous()
        weight = weight.contiguous()

        ctx.cols = cols
        ctx.rows = x.numel() // cols
        if x.numel() % cols != 0 or cols % 8 != 0:
            raise ValueError(f"Input size {x.numel()} not divisible by cols {cols} or unsupported cols value (must to be divisible by 8)")
        ctx.eps = epsilon

        output = torch.empty_like(x)
        invvar = torch.empty(ctx.rows, device=x.device, dtype=torch.float32) 

        # Launch optimized CUDA kernel
        custom_rmsnorm.torch_launch_rms_forward(
            output, x, weight, ctx.rows, ctx.cols, ctx.eps, invvar
        )

        ctx.save_for_backward(output, weight, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for RMSNorm operation.
        
        Args:
            ctx: Context containing saved tensors from forward pass
            grad_output: Gradient of loss 
            
        Returns:
            grad_input: Gradient for input x
            grad_weight: Gradient for weight parameter
            None: Placeholder for epsilon gradient
            None: Placeholder for cols gradient
        
        """
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        fwd_pass_output, weight, invvar = ctx.saved_tensors
        

        grad_input = torch.empty_like(fwd_pass_output)  
        grad_weight = torch.empty_like(weight)     

        # Launch optimized CUDA backward kernel
        custom_rmsnorm.torch_launch_rms_backward(
            grad_input, grad_weight,
            grad_output, 
            fwd_pass_output,
            weight, invvar, ctx.rows, ctx.cols
        )

        return grad_input, grad_weight, None, None
    
class Fused_RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
       # init.ones_(self.weight)

        

    def forward(self, x):

        cols = self.hidden_size
        if x.shape[-1] != cols:
             raise ValueError(f"Last dim of input x must be hidden_size {cols}, got {x.shape[-1]}")
        # Input x can be various dtypes, e.g. bfloat16. Weight is bfloat16.
        return RMSNormModelFunction.apply(x, self.weight, self.eps, cols)