import os 
import torch
import torch.nn.functional as F
from typing import Tuple, Callable
from unittest import TestCase
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

class HunyuanParams:
    hidden_size: int = 3072
    activation_func: Callable = F.gelu
    add_qkv_bias: bool = True
    in_channels: int = 33
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 1
    num_single_layers: int = 2
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 2
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    guidance_embeds: bool = True
    text_embed_dim: int = 4096
    pooled_projection_dim: int = 768
    rope_theta: float = 256.0
    rope_axes_dim: Tuple[int] = (16, 56, 56)


HUNYUAN_MODEL_FWD_SUCCESS = "Parallel Hunyuan model forward test success"
HUNYUAN_MODEL_FWD_FAIL = "Parallel Hunyuan model forward test fail"
HUNYUAN_MODEL_BWD_SUCCESS = "Parallel Hunyuan model backward test success"
HUNYUAN_MODEL_BWD_FAIL = "Parallel Hunyuan model backward test fail"

@patch("teletron.utils.get_args")
def parallel_hunyuan_model_testing(rank, world_size, q, mock_teletron):
    from teletron.models.hunyuan import HunyuanVideoTransformer3DModel, ParallelHunyuanVideoModel
    from teletron.core.parallel_state import initialize_model_parallel_base
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.activation_offload = True
    args.num_layers = 1 
    args.num_attention_heads = 2
    args.distributed_vae = False
    args.consumer_models_num = 1
    mock_teletron.return_value = args


    cp_size = world_size
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    initialize_model_parallel_base(
            tensor_model_parallel_size = 1,
            pipeline_model_parallel_size = 1,
            virtual_pipeline_model_parallel_size = None,
            pipeline_model_parallel_split_rank = None,
            use_sharp = False,
            context_parallel_size = cp_size,
            expert_model_parallel_size = 1,
            nccl_communicator_config_path = None,
            distributed_timeout_minutes = 30,
        )
    hunyuanConfig = HunyuanParams()
    torch.manual_seed(1234)
    hunyuan_model = HunyuanVideoTransformer3DModel(hunyuanConfig).cuda().to(torch.bfloat16)
    torch.manual_seed(1234)
    parallel_hunyuan_model = ParallelHunyuanVideoModel(hunyuanConfig).cuda().to(torch.bfloat16)
    # test forward
    parallel_hunyuan_model.load_state_dict(hunyuan_model.state_dict())

    # from tensorwatch import watch_module_forward_backward, TensorWatch
    # watch_module_forward_backward(parallel_hunyuan_model)

    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/qiuyang/datasets/rank0_inputs.pt", map_location=f"cuda:{rank}")
    hunyuan_model_output = hunyuan_model(**input_dict)

    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/qiuyang/datasets/rank0_inputs.pt", map_location=f"cuda:{rank}")
    parallel_hunyuan_model_output = parallel_hunyuan_model(**input_dict)

    if is_close_by_normalized_euclid_dist(hunyuan_model_output[0], parallel_hunyuan_model_output[0]):
        q.put(f"{HUNYUAN_MODEL_FWD_SUCCESS} rank{rank}")
    else:
        logging.info(f"normalized_euclid_dist {normalized_euclid_dist(hunyuan_model_output[0], parallel_hunyuan_model_output[0])} {hunyuan_model_output[0].norm()} {parallel_hunyuan_model_output[0].norm()} rank{rank}")
        q.put(f"{HUNYUAN_MODEL_FWD_FAIL} rank{rank}")


    # test backward
    hunyuan_model_output[0].backward(torch.ones_like(hunyuan_model_output[0]))
    parallel_hunyuan_model_output[0].backward(torch.ones_like(parallel_hunyuan_model_output[0]))

    # TensorWatch.step()
    model_grads = {name: param.grad for name, param in hunyuan_model.named_parameters() if param.grad is not None}
    parallel_moedl_grads = {name: param.grad for name, param in parallel_hunyuan_model.named_parameters() if param.grad is not None}
    grad_allclose = True
    for name in model_grads:
        norm_euclid_dist = normalized_euclid_dist(model_grads[name], parallel_moedl_grads[name])
        if norm_euclid_dist < 0.02:
            continue
        else:
            logging.info(f"{name}: {norm_euclid_dist} {model_grads[name].norm()} {parallel_moedl_grads[name].norm()} rank{rank}")
            if model_grads[name].norm() < 2e-4:
                continue
            grad_allclose = False
    if grad_allclose:
        q.put(f"{HUNYUAN_MODEL_BWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{HUNYUAN_MODEL_BWD_FAIL} rank{rank}")

def normalized_euclid_dist(output, parallel_output):
    hunyuan_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (hunyuan_norm + parallel_norm)
    return normalized_euclid_dist
    
def is_close_by_normalized_euclid_dist(output, parallel_output):
    hunyuan_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (hunyuan_norm + parallel_norm)
    if normalized_euclid_dist < 0.001:
        return True 
    else:
        return False 

class testParallelHunyuanVideoModel(TestCase):
    def test_forward_backward(self):
        cp_size = 2
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(cp_size, parallel_hunyuan_model_testing)
        correct_responses = [f"{HUNYUAN_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        correct_responses += [f"{HUNYUAN_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        responses = []
        while not q.empty():
            res = q.get()
            responses.append(res)
        self.assertEqual(sorted(responses), correct_responses)
        #TODO: test backward