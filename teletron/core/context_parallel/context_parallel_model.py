import torch
import torch.nn as nn 
from megatron.core import mpu
from teletron.core.context_parallel.mappings import split_forward_gather_backward,\
        gather_forward_split_backward


class ContextParallelModelManager():
    def __init__(self, split_dim=1, gather_dim=1):
        self.cp_size = mpu.get_context_parallel_world_size()
        self.cp_group = mpu.get_context_parallel_group()
        self.split_dim = split_dim
        self.gather_dim = gather_dim 
        self.use_pad = None
    
    def split_input(self, x):
        # assert x is not parallel
        if x.shape[self.split_dim] % self.cp_size != 0 :
            self.origin_length = x.shape[self.split_dim]
            self.padded_length = self.origin_length + self.cp_size - \
                (self.origin_length % self.cp_size)
            x = self.pad_for_context_parallel(x)
            self.use_pad = True
        else:
            self.use_pad = False
            
        x = split_forward_gather_backward(x, self.cp_group, dim=self.split_dim, grad_scale="none")
        return x

    def gather_output(self, output):
        output = gather_forward_split_backward(output, self.cp_group, dim=self.gather_dim, grad_scale="none")
        if self.use_pad:
            output = self.remove_pad_for_context_parallel(output)
        return output 

    def pad_for_context_parallel(self, tensor):
        pad_size = int(self.padded_length - self.origin_length)

        if pad_size <= 0:
            return tensor  # No padding needed

        # Create pad tuple: (dim_n_before, dim_n_after, ..., dim_0_before, dim_0_after)
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * self.split_dim + 1)] = pad_size  # pad after the dimension
        return torch.nn.functional.pad(tensor, pad) 
    
    def remove_pad_for_context_parallel(self, tensor):
        return tensor.narrow(self.gather_dim, 0, self.origin_length)

    # def context_parallel_forward_transformer_blocks(self, forward_func):
    #     @wraps(forward_func)
    #     def cp_forward_func(cp_args=[], cp_kwargs={}, regular_args=[], regular_kwargs={}):

            

