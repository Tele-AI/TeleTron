import os 
import torch
from unittest import TestCase
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
import logging
from teletron.models.longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from teletron.models.longcat_video.parallel_longcat_model import ParallelLongCatModel

logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

LONGCAT_MODEL_FWD_SUCCESS = "Parallel LongCat model forward test success"
LONGCAT_MODEL_FWD_FAIL = "Parallel LongCat model forward test fail"
LONGCAT_MODEL_BWD_SUCCESS = "Parallel LongCat model backward test success"
LONGCAT_MODEL_BWD_FAIL = "Parallel LongCat model backward test fail"

def normalized_euclid_dist(output, parallel_output):
    model_norm = output.norm().item()
    parallel_norm = parallel_output.norm().item()
    euclid_dist = torch.norm(output - parallel_output)
    normalized_euclid_dist = 0.5 * euclid_dist / (model_norm + parallel_norm + 1e-6)
    return normalized_euclid_dist

def is_close_by_normalized_euclid_dist(output, parallel_output):
    dist = normalized_euclid_dist(output, parallel_output)
    if dist < 0.001:
        return True 
    else:
        return False 

@patch("teletron.models.longcat_video.parallel_longcat_model.set_config")
@patch("teletron.utils.get_args")
def parallel_longcat_model_testing(rank, world_size, q, mock_get_args, mock_set_config):
    # Setup distributed
    cp_size = world_size
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # Mock args
    args = Mock()
    args.activation_offload = False
    mock_get_args.return_value = args

    # Model Params
    hidden_size = 64
    num_heads = 4
    depth = 2
    patch_size = (1, 2, 2)
    in_channels = 4
    out_channels = 4
    text_dim = 32
    mlp_ratio = 4.0
    
    # Config for ParallelLongCatModel
    dit_config = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        caption_channels=text_dim,
        mlp_ratio=mlp_ratio,
        adaln_tembed_dim=32,
        frequency_embedding_size=32,
        patch_size=patch_size,
        enable_flashattn3=False,
        enable_flashattn2=False,
        enable_xformers=False,
        enable_bsa=False,
        bsa_params=None,
        cp_split_hw=[1, cp_size], # Split on Width
        text_tokens_zero_pad=False,
    )

    mock_config = {
        'model_config': {
            'dit': {
                'config': dit_config
            }
        }
    }
    mock_set_config.return_value = mock_config

    # Initialize Models
    torch.manual_seed(1234)
    model = LongCatVideoTransformer3DModel(**dit_config).cuda().to(torch.bfloat16)
    
    torch.manual_seed(1234)
    parallel_model = ParallelLongCatModel(config=None).cuda().to(torch.bfloat16)
    
    # Load weights
    parallel_model.load_state_dict(model.state_dict())

    # Inputs
    B = 2
    T = 4
    H = 8
    W = 8 * cp_size 
    
    hidden_states = torch.randn(B, in_channels, T, H, W).cuda().to(torch.bfloat16).requires_grad_(True)
    timestep = torch.randint(0, 100, (B,)).cuda().to(torch.bfloat16)
    encoder_hidden_states = torch.randn(B, 1, 10, text_dim).cuda().to(torch.bfloat16).requires_grad_(True)
    encoder_attention_mask = torch.ones(B, 1, 10, 1).cuda().to(torch.bfloat16) # [B, 1, L, 1]

    # Forward
    out = model(
        hidden_states, 
        timestep, 
        encoder_hidden_states, 
        encoder_attention_mask
    )
    
    p_hidden_states = hidden_states.clone().detach().requires_grad_(True)
    p_encoder_hidden_states = encoder_hidden_states.clone().detach().requires_grad_(True)
    
    p_out = parallel_model(
        p_hidden_states, 
        timestep, 
        p_encoder_hidden_states, 
        encoder_attention_mask
    )

    # Check Forward
    if is_close_by_normalized_euclid_dist(out, p_out):
        q.put(f"{LONGCAT_MODEL_FWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{LONGCAT_MODEL_FWD_FAIL} rank{rank}")
        
    # Backward
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    p_out.backward(grad_output)

    # Check Gradients
    grad_allclose = True
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        p_param = dict(parallel_model.named_parameters())[name]
        if p_param.grad is None:
            grad_allclose = False
            logging.error(f"Grad missing for {name} in parallel model")
            break
            
        dist = normalized_euclid_dist(param.grad, p_param.grad)
        if dist >= 0.02: # Threshold from reference
            grad_allclose = False
            logging.info(f"Grad mismatch {name}: {dist}")

    if grad_allclose:
        q.put(f"{LONGCAT_MODEL_BWD_SUCCESS} rank{rank}")
    else:
        q.put(f"{LONGCAT_MODEL_BWD_FAIL} rank{rank}")

class testParallelLongCatModel(TestCase):
    def test_forward_backward(self):
        if torch.cuda.device_count() < 2:
            print("Skipping CP test on single GPU")
            return
        cp_size = 2
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12555' 
        q = spawn(cp_size, parallel_longcat_model_testing)
        
        correct_responses = []
        correct_responses += [f"{LONGCAT_MODEL_FWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        correct_responses += [f"{LONGCAT_MODEL_BWD_SUCCESS} rank{rank}" for rank in range(cp_size)]
        
        responses = []
        while not q.empty():
            res = q.get()
            responses.append(res)
        
        self.assertEqual(sorted(responses), sorted(correct_responses))
