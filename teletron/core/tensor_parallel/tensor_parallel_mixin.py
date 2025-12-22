from .layers import TeleColumnParallelLinear, TeleRowParallelLinear, TeleParallelRMSNorm

from megatron.core import mpu

import torch.nn as nn

class TensorParallelMixin:
        
    def enable_col_parallel(
        self, 
        linear_module: nn.Module,
        config,
        bias: bool=True,
        skip_bias_add: bool=False,
        gather_output: bool=False,
        skip_weight_param_allocation: bool=False,
        ):
        """thank you nvidia"""
        return TeleColumnParallelLinear(
                linear_module.in_features,
                linear_module.out_features,
                config=config,
                init_method=config.init_method,
                bias=bias,
                skip_bias_add=skip_bias_add,
                gather_output=gather_output,
                skip_weight_param_allocation=skip_weight_param_allocation,
                )  

            
    def enable_row_parallel(
        self, 
        linear_module: nn.Module,
        config,
        bias: bool=True,
        input_is_parallel: bool=True,
        skip_bias_add: bool=False,
        ):
        """thank you nvidia again."""
        return TeleRowParallelLinear(
                linear_module.in_features,
                linear_module.out_features,
                config=config,
                init_method=config.init_method,
                bias=bias,
                input_is_parallel=input_is_parallel,
                skip_bias_add=skip_bias_add,
                )
    
    def enable_rms_norm_parallel(self, rmsnorm_module: nn.Module, dim):
        """special cuisine on rmsnorm"""
        return TeleParallelRMSNorm( dim = dim, eps = rmsnorm_module.eps)
    
    def enable_ffn_tensor_parallel(self, ffn_module, config):
        """enable ffn layer's tensor_parallel."""
        world_size = mpu.get_tensor_model_parallel_world_size()
        if world_size == 1:
            return
        ffn_module[0] = self.enable_col_parallel(ffn_module[0], config=config)
        ffn_module[2] = self.enable_row_parallel(ffn_module[2], config=config)

    def enable_self_attn_tensor_parallel(self, module: nn.Module, config):
        """enable self attention layer's tensor parallel"""
        world_size = mpu.get_tensor_model_parallel_world_size()
        if world_size == 1:
            return
        module.num_heads = module.num_heads // world_size
        
        module.query = self.enable_col_parallel(module.query, config=config, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, config=config, gather_output=False)
        module.norm_query = self.enable_rms_norm_parallel(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel(module.norm_key, module.dim)
        
        module.value = self.enable_col_parallel(module.value, config=config, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj, config=config)
        module.attn.num_heads = module.num_heads
    
    
    def enable_cross_attn_tensor_parallel(self, module: nn.Module, config):
        """enable cross attention layer's tensor parallel"""
        world_size = mpu.get_tensor_model_parallel_world_size()
        if world_size == 1:
            return
        
        module.num_heads = module.num_heads // world_size

        module.query = self.enable_col_parallel(module.query, config=config, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, config=config,  gather_output=False)
        module.value = self.enable_col_parallel(module.value, config=config, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj, config=config)
        module.norm_query = self.enable_rms_norm_parallel(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel(module.norm_key, module.dim)
        
        if module.has_image_input:
            module.img_key = self.enable_col_parallel(module.img_key, config=config,  gather_output=False) 
            module.img_value = self.enable_col_parallel(module.img_value, config=config, gather_output=False)
            module.norm_image_key = self.enable_rms_norm_parallel(module.norm_image_key, module.dim)
            
        module.attn.num_heads = module.num_heads
        module.attn2.num_heads = module.num_heads
        