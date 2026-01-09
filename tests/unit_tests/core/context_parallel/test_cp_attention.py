
import os 
from unittest import TestCase
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu
from einops import rearrange
from teletron.core.context_parallel.mappings import SeqAllToAll, set_global_all_to_all_buffer, split_forward_gather_backward, gather_forward_split_backward
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = rearrange(q, "b s (n d) -> b s n d", n=40)
        k = rearrange(k, "b s (n d) -> b s n d", n=40)
        v = rearrange(v, "b s (n d) -> b s n d", n=40)
        if FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_func(q, k, v)[0]
            x = x.transpose(1, 2).contiguous()
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h
        o = self.o_linear(x)
        return o

class CPAttn(Attn, ):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        for name, param in self.named_parameters():
            param.register_hook(self.cp_grad_reduce)
    
    @staticmethod   
    def cp_grad_reduce(grad):
        with torch.no_grad():
            cp_size = mpu.get_context_parallel_world_size()
            dim_size = list(grad.size())
            dim_size[0] = dim_size[0] * cp_size
            grad_list = torch.empty(dim_size, dtype=grad.dtype, device=torch.cuda.current_device())
            torch.distributed._all_gather_base(grad_list, grad.contiguous(), group=mpu.get_context_parallel_group())
            grad_list = grad_list.view(cp_size, -1, *grad_list.shape[1:])
            reduced_grad = grad_list.sum(dim=0)
            # # allreduce
            # reduced_grad = grad.contiguous()
            # torch.distributed.all_reduce(reduced_grad, group=mpu.get_context_parallel_group())
        return reduced_grad

        
    def forward(self, x):
        cp_group = mpu.get_context_parallel_group()
        x = split_forward_gather_backward(x, cp_group, dim=1, grad_scale="none")
        set_global_all_to_all_buffer(x.numel())
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = rearrange(q, "b s (n d) -> b s n d", n=40)
        k = rearrange(k, "b s (n d) -> b s n d", n=40)
        v = rearrange(v, "b s (n d) -> b s n d", n=40)
        if mpu.get_context_parallel_world_size() > 1:
            q = SeqAllToAll.apply(cp_group, q, 2, 1, True)
            k = SeqAllToAll.apply(cp_group, k, 2, 1, True)
            v = SeqAllToAll.apply(cp_group, v, 2, 1, True)
            # qkv: b s n/CP d

        if FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_func(q, k, v)[0]
            x = x.transpose(1, 2).contiguous()
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            x = F.scaled_dot_product_attention(q, k, v)
        if mpu.get_context_parallel_world_size() > 1:
            x = SeqAllToAll.apply(
                cp_group, x, 2, 1, True
            )  # b img_seq sub_n d
            # x: b n s/CP d
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h
        o = self.o_linear(x)
        o = gather_forward_split_backward(o, cp_group, dim=1, grad_scale="none")
        return o


@patch("teletron.utils.get_args")
def buffer_test_func(rank, world_size, q, mock_teletron):
    from teletron.core.parallel_state import initialize_model_parallel_base 
    args = Mock()
    args.consumer_models_num = 1
    mock_teletron.return_value = args

    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    
    initialize_model_parallel_base(
        tensor_model_parallel_size = 1,
        pipeline_model_parallel_size = 1,
        virtual_pipeline_model_parallel_size = None,
        pipeline_model_parallel_split_rank = None,
        use_sharp = False,
        context_parallel_size = world_size,
        expert_model_parallel_size = 1,
        nccl_communicator_config_path = None,
        distributed_timeout_minutes = 30,
    )

    torch.manual_seed(1234)
    x = torch.randn(1, 4096, 5120, dtype=torch.bfloat16).cuda()
    
    attn = Attn(5120).cuda().to(torch.bfloat16)
    cp_attn = CPAttn(5120).cuda().to(torch.bfloat16)
    cp_attn.load_state_dict(attn.state_dict())

    output = attn(x)
    cp_output = cp_attn(x)
    

    if is_close_by_normalized_euclid_dist(output, cp_output):
        q.put(f"cp attn forward success rank{rank}")
    else:
        q.put(f"cp attn forward failed rank{rank}")

    output.backward(torch.ones_like(output))
    cp_output.backward(torch.ones_like(cp_output))

    model_grads = {name: param.grad for name, param in attn.named_parameters() if param.grad is not None}
    parallel_model_grads = {name: param.grad for name, param in cp_attn.named_parameters() if param.grad is not None}
    grad_allclose = True
    for name in model_grads:
        norm_euclid_dist = normalized_euclid_dist(model_grads[name], parallel_model_grads[name])
        logging.info(f"{name}: {norm_euclid_dist} {model_grads[name].norm()} {parallel_model_grads[name].norm()} rank{rank}")
        if norm_euclid_dist < 0.02 or (norm_euclid_dist > 0.02 and model_grads[name].norm() < 100):
            continue
        else:
            grad_allclose = False
    if grad_allclose:
        q.put(f"cp attn backward success rank{rank}")
    else:
        q.put(f"cp attn backward failed rank{rank}")

def normalized_euclid_dist(output, parallel_output):
    wan_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (wan_norm + parallel_norm)
    return normalized_euclid_dist

def is_close_by_normalized_euclid_dist(output, parallel_output):
    wan_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (wan_norm + parallel_norm)
    if normalized_euclid_dist < 0.001:
        return True 
    else:
        return False 

class testcpattn(TestCase):
    def test_forward_backward(self):
        world_size = 2
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(world_size, buffer_test_func)
        correct_responses = [f"cp attn backward success rank{rank}" for rank in range(world_size)]
        correct_responses += [f"cp attn forward success rank{rank}" for rank in range(world_size)]
        responses = []
        while not q.empty():
            res = q.get()
            responses.append(res)
        self.assertEqual(sorted(responses), correct_responses)
        #TODO: test backward

