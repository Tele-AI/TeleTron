import os 
import torch
from unittest import TestCase
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
import logging

logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

class WanParams:
    num_attention_heads: int = 40
    hidden_size: int = 5120
    num_layers: int = 1


WAN_MODEL_FWD_SUCCESS = "Parallel Wan model forward test success"
WAN_MODEL_FWD_FAIL = "Parallel Wan model forward test fail"
WAN_MODEL_BWD_SUCCESS = "Parallel Wan model backward test success"
WAN_MODEL_BWD_FAIL = "Parallel Wan model backward test fail"

@patch("teletron.utils.get_args")
def parallel_wan_model_testing(rank, world_size, q, mock_teletron):
    from teletron.models.wan import ParallelWanModel, WanModel
    from teletron.core.parallel_state import initialize_model_parallel_base 
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.activation_offload = True
    args.num_layers = 1 
    args.num_attention_heads = 40
    args.distributed_vae = False
    mock_teletron.return_value = args
    args.consumer_models_num = 1

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
    wanConfig = WanParams()
    torch.manual_seed(1234)
    wan_model = WanModel(wanConfig).cuda().to(torch.bfloat16)
    torch.manual_seed(1234)
    parallel_wan_model = ParallelWanModel(wanConfig).cuda().to(torch.bfloat16)

    parallel_wan_model.load_state_dict(wan_model.state_dict())
    # wan_params = dict(wan_model.named_parameters())
    # wan_parallel_params = dict(parallel_wan_model.named_parameters())

    # from tensorwatch import watch_module_forward_backward, TensorWatch
    # watch_module_forward_backward(parallel_wan_model)

    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/test/test_data/transformer_inputs.pt", map_location=f"cuda:{rank}")
    wan_model_output = wan_model(**input_dict)
    input_dict = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/test/test_data/transformer_inputs.pt", map_location=f"cuda:{rank}")
    parallel_wan_model_output = parallel_wan_model(**input_dict)
    if is_close_by_normalized_euclid_dist(wan_model_output, parallel_wan_model_output):
        q.put(f"{WAN_MODEL_FWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{WAN_MODEL_FWD_FAIL} rank{rank}")
    #TODO: test backward
    # test backward
    wan_model_output.backward(torch.ones_like(wan_model_output))
    parallel_wan_model_output.backward(torch.ones_like(parallel_wan_model_output))
    # TensorWatch.step()
    model_grads = {name: param.grad for name, param in wan_model.named_parameters() if param.grad is not None}
    parallel_model_grads = {name: param.grad for name, param in parallel_wan_model.named_parameters() if param.grad is not None}
    grad_allclose = True
    for name in model_grads:
        norm_euclid_dist = normalized_euclid_dist(model_grads[name], parallel_model_grads[name])
        logging.info(f"{name}: {norm_euclid_dist} {model_grads[name].norm()} {parallel_model_grads[name].norm()} rank{rank}")
        if norm_euclid_dist < 0.02:
            continue
        else:
            grad_allclose = False
    if grad_allclose:
        q.put(f"{WAN_MODEL_BWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{WAN_MODEL_BWD_FAIL} rank{rank}")
    


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



class testParallelWanModel(TestCase):
    def test_forward_backward(self):
        cp_size = 2
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(cp_size, parallel_wan_model_testing)
        correct_responses = [f"{WAN_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        correct_responses += [f"{WAN_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        responses = []
        while not q.empty():
            res = q.get()
            responses.append(res)
        self.assertEqual(sorted(responses), correct_responses)
        #TODO: test backward

