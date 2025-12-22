import os 
from dataclasses import dataclass, asdict
import torch
from typing import Tuple
from unittest import TestCase
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
from megatron.core import mpu
import logging

TELEAI_MODEL_FWD_SUCCESS = "Parallel Wan model forward test success"
TELEAI_MODEL_FWD_FAIL = "Parallel Wan model forward test fail"
TELEAI_MODEL_BWD_SUCCESS = "Parallel Wan model backward test success"
TELEAI_MODEL_BWD_FAIL = "Parallel Wan model backward test fail"

CUDA_DEVICES = [0, 1, 2, 3]

@dataclass
class TeleaiParams:
    dim: int = 5120
    in_dim: int = 36
    out_dim: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    eps: float = 1e-6
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_heads: int = 40
    num_layers: int = 1
    has_image_input: bool = True
    has_image_pos_emb: bool = False

@patch("teletron.utils.set_config")
@patch("teletron.utils.get_args")
def parallel_teleai_model_testing(rank, world_size, q, tp_size, cp_size, mock_get_args, mock_set_config):
    from teletron.models.teleai import ParallelTeleaiModel,TeleaiModel
    
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.activation_offload = True
    args.num_layers = 1 
    args.num_attention_heads = 40
    args.distributed_vae = False
    args.consumer_models_num = 1
    mock_get_args.return_value = args
    
    model_config=dict(
        dit=dict(
            type="ParallelTeleaiModel", # ParallelTeleaiModel
            config=dict(
                has_image_input=True, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36, # t2v:16 i2v:36
                dim=5120, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40, # 1.3B:12 10B:40 14B:40
                num_layers=1, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=False, 
            ),
        ))
    mock_set_config.return_value = {"model_config":model_config}
    

    from megatron.core.transformer import TransformerConfig
    cfg = Mock(spec=TransformerConfig)
    cfg._cpu_offloading_context = None
    cfg.perform_initialization = True
    cfg.use_cpu_initialization = True
    cfg.params_dtype = torch.bfloat16
    cfg.gradient_accumulation_fusion = False
    cfg.expert_model_parallel_size = 1 
    cfg.defer_embedding_wgrad_compute = False
    cfg.async_tensor_model_parallel_allreduce = False
    cfg.num_layers = args.num_layers
    cfg.sequence_parallel = False
    
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    
    global CUDA_DEVICES
    assert len(CUDA_DEVICES) >= world_size, f"GPU number is not enough"
    cuda_rank = CUDA_DEVICES[rank]
    torch.cuda.set_device(cuda_rank)
    
    from teletron.core.parallel_state import initialize_model_parallel_base
    initialize_model_parallel_base(
            tensor_model_parallel_size = tp_size,
            pipeline_model_parallel_size = 1,
            virtual_pipeline_model_parallel_size = None,
            pipeline_model_parallel_split_rank = None,
            use_sharp = False,
            context_parallel_size = cp_size,
            expert_model_parallel_size = 1,
            nccl_communicator_config_path = None,
            distributed_timeout_minutes = 30,
        )
    
    teleaiConfig = TeleaiParams()
    torch.manual_seed(1234)
    
    teleai_model = TeleaiModel(**asdict(teleaiConfig)).cuda(cuda_rank).to(torch.bfloat16)
    torch.manual_seed(1234)
    parallel_teleai_model = ParallelTeleaiModel(cfg).cuda(cuda_rank).to(torch.bfloat16)
    
    parallel_teleai_model.load_state_dict(tp_load_state_dict(teleai_model))
    
    input_dict = torch.load("/nvfile-heatstorage/ai_infra/data/lit117/teletron-testing/test_data/saved_inputs_360/input_dict_iter0_rank0.pt", map_location=f"cuda:{cuda_rank}")

    teleai_model_output = teleai_model(x=input_dict['noisy_latents'],
                                    timestep=input_dict['timestep'],
                                    context=input_dict['prompt_emb']['context'],
                                    clip_feature = input_dict['image_emb']['clip_feature'],
                                    y=input_dict['image_emb']['y'])  
    
    parallel_teleai_model_output = parallel_teleai_model(x=input_dict['noisy_latents'],
                                    timestep=input_dict['timestep'],
                                    context=input_dict['prompt_emb']['context'],
                                    clip_feature = input_dict['image_emb']['clip_feature'],
                                    y=input_dict['image_emb']['y'])  
        
    if is_close_by_normalized_euclid_dist(teleai_model_output, parallel_teleai_model_output):
        q.put(f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{TELEAI_MODEL_FWD_FAIL} rank{rank}")

    teleai_model_output.backward(torch.ones_like(teleai_model_output))
    parallel_teleai_model_output.backward(torch.ones_like(parallel_teleai_model_output))
    model_grads = {name: param.grad for name, param in teleai_model.named_parameters() if param.grad is not None}
    parallel_model_grads = {name: param.grad for name, param in parallel_teleai_model.named_parameters() if param.grad is not None}
    grad_allclose = True
    
    tp_rank = mpu.get_tensor_model_parallel_rank()
    for name in model_grads:
        norm_euclid_dist = tp_normalized_euclid_dist(tp_rank, name, model_grads[name], parallel_model_grads[name])
        if norm_euclid_dist < 0.02:
            continue
        else:
            logging.info(f"{name}: {norm_euclid_dist} {model_grads[name].norm().item()} {parallel_model_grads[name].norm().item()} rank{rank}")
            grad_allclose = False
    if grad_allclose:
        q.put(f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{TELEAI_MODEL_BWD_FAIL} rank{rank}")


def tp_normalized_euclid_dist(rank, name, output, parallel_output):    
    col_w = ["self_attn.query.weight", "self_attn.key.weight", "self_attn.value.weight","ffn.0.weight",
             "cross_attn.query.weight", "cross_attn.key.weight", "cross_attn.value.weight",
             "cross_attn.img_key.weight", "cross_attn.img_value.weight"]
    
    col_b = ["self_attn.query.bias", "self_attn.key.bias", "self_attn.value.bias","ffn.0.bias",
             "cross_attn.query.bias", "cross_attn.key.bias", "cross_attn.value.bias",
             "cross_attn.img_key.bias", "cross_attn.img_value.bias"]
    
    row_w = ["ffn.2.weight", "self_attn.out_proj.weight",
             "cross_attn.out_proj.weight"]
    
    norm_w = ["self_attn.norm_query.weight", "self_attn.norm_key.weight",
              "cross_attn.norm_query.weight", "cross_attn.norm_key.weight",
              "cross_attn.norm_image_key.weight"]
    
    def normalized_euclid_dist(rank, name, output, parallel_output):
        teleai_norm = output.norm().item()
        parallel_norm = parallel_output.norm().item()
        euclid_dist = torch.norm(output - parallel_output)
        normalized_euclid_dist = 0.5 * euclid_dist / (teleai_norm + parallel_norm)
        logging.info(f"{name}: {normalized_euclid_dist} {output.norm().item()} {parallel_output.norm().item()} rank{rank}")
        return normalized_euclid_dist
            
    if any(cw in name for cw in col_w):
        size = parallel_output.shape[0]
        return normalized_euclid_dist(rank, name, output[rank*size: (rank+1)*size, :], parallel_output)
    elif any(cb in name for cb in col_b):
        
        size = parallel_output.shape[0]
        return normalized_euclid_dist(rank, name, output[rank*size: (rank+1)*size], parallel_output)
    elif any(rw in name for rw in row_w):
        size = parallel_output.shape[1]
        return normalized_euclid_dist(rank, name, output[:, rank*size: (rank+1)*size], parallel_output)
    elif any(nw in name for nw in norm_w):
        size = parallel_output.shape[0]
        return normalized_euclid_dist(rank, name, output[rank*size:(rank+1)*size], parallel_output)
    else:
        return normalized_euclid_dist(rank, name, output, parallel_output)

def is_close_by_normalized_euclid_dist(output, parallel_output):
    teleai_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (teleai_norm + parallel_norm)
    if normalized_euclid_dist < 0.001:
        return True 
    else:
        return False 
    
def tp_load_state_dict(base_model):
    
    base_dict = base_model.state_dict()
    tp_dict = {}
    
    col_w = ["self_attn.query.weight", "self_attn.key.weight", "self_attn.value.weight","ffn.0.weight",
             "cross_attn.query.weight", "cross_attn.key.weight", "cross_attn.value.weight",
             "cross_attn.img_key.weight", "cross_attn.img_value.weight"]
    
    col_b = ["self_attn.query.bias", "self_attn.key.bias", "self_attn.value.bias","ffn.0.bias",
             "cross_attn.query.bias", "cross_attn.key.bias", "cross_attn.value.bias",
             "cross_attn.img_key.bias", "cross_attn.img_value.bias"]
    
    row_w = ["ffn.2.weight", "self_attn.out_proj.weight",
             "cross_attn.out_proj.weight"]
    
    norm_w = ["self_attn.norm_query.weight", "self_attn.norm_key.weight",
              "cross_attn.norm_query.weight", "cross_attn.norm_key.weight",
              "cross_attn.norm_image_key.weight"]
    
    def tp_col_weight_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        
        size = param.shape[0] // tp_size
        tp_dict[name] = param[rank*size:(rank+1)*size,:]

    def tp_col_bias_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        
        size = param.shape[0] // tp_size
        tp_dict[name] = param[rank*size:(rank+1)*size]

    def tp_row_weight_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        
        size = param.shape[1] // tp_size
        tp_dict[name] = param[:, rank*size:(rank+1)*size]

    def tp_row_bias_load(tp_dict, name, param):
        tp_dict[name] = param
        
    def tp_norm_weight_load(tp_dict, name, param):
        rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        size = param.shape[0] // tp_size
        tp_dict[name] = param[rank*size:(rank+1)*size]
    
    for name, param in base_dict.items():
        if any(cw in name for cw in col_w):
            tp_col_weight_load(tp_dict, name, param)
        elif any(cb in name for cb in col_b):
            tp_col_bias_load(tp_dict, name, param)
        elif any(rw in name for rw in row_w):
            tp_row_weight_load(tp_dict, name, param)
        elif any(nw in name for nw in norm_w):
            tp_norm_weight_load(tp_dict, name, param)
        else:
            tp_dict[name] = param
    return tp_dict


def launch_multiprocess_testing(world_size, tp_size, cp_size):
    assert world_size == tp_size * cp_size
    os.environ['WORLD_SIZE'] = str(world_size )
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12445'
    q = spawn(world_size, parallel_teleai_model_testing, tp_size, cp_size)

    responses = []
    while not q.empty():
        res = q.get()
        responses.append(res)
    return responses

class testParallelWanModel(TestCase):
    def test_tp(self):
        world_size = tensor_model_parallel_world_size = 2
        responses = launch_multiprocess_testing(world_size, tensor_model_parallel_world_size, 1)

        correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size )]
        correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size )]
        
        self.assertEqual(sorted(responses), correct_responses)
        
    def test_cp(self):
        world_size = cp_size = 2
        responses = launch_multiprocess_testing(world_size, 1, cp_size)

        correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size )]
        correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size )]
        
        self.assertEqual(sorted(responses), correct_responses)

    # def test_tp_cp(self):
    #     cp_size = 2
    #     tp_size = 2
    #     world_size = cp_size * tp_size
    #     responses = launch_multiprocess_testing(world_size, tp_size, cp_size)

    #     correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size )]
    #     correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size )]
        
    #     self.assertEqual(sorted(responses), correct_responses)
        
    
if __name__ == "__main__":
    tensor_model_parallel_world_size = 2
    world_size = 4
    cp_size = world_size // tensor_model_parallel_world_size
    responses = launch_multiprocess_testing(world_size, tensor_model_parallel_world_size, cp_size)
    
    correct_responses = [f"{TELEAI_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(world_size )]
    correct_responses += [f"{TELEAI_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(world_size )]
    print(f"test_result: {responses}")
    assert sorted(responses) == correct_responses
    print("test success!")