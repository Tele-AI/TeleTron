import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer import TransformerConfig
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from teletron.utils import (get_args,
                            fused_kernel_load,
                            print_rank_0,
                            print_rank_last,
                            get_num_microbatches,
                            update_num_microbatches,
                            get_attr_wrapped_model,
                            get_model_config,
                            get_current_global_batch_size
                            )
import time
import contextlib
from datetime import timedelta
from megatron.core.jit import jit_fuser
import torch.distributed as dist
from typing import Optional
import dataclasses
import torch.nn.functional as F
import random
import numpy as np
from datetime import datetime
import math

from apex.multi_tensor_apply import multi_tensor_applier

try:
    import amp_C
except ImportError:
    amp_C = None
NUM_BYTES_IN_MEGABYTE = 1024 * 1024
_TRANSFORMER_MODEL_GROUP = None



def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)
        
def compute_weight_and_optimizer_memory(args, verbose=False):
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
            + (args.num_query_groups / args.num_attention_heads)
            + (2 / args.hidden_size)
            + (1 / (args.num_layers * args.hidden_size))
        )
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: {num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: {num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: {num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: {num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this function
    # are for the first pipeline stage.

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / args.tensor_model_parallel_size


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not args.sequence_parallel or args.recompute_granularity != 'selective':
        return

    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )
    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, "
        f"total={total_memory:.2f} MB\n"
    )

def get_grad_norm(optimizer):
    parameters = optimizer.get_parameters()
    grads_for_norm = optimizer.get_main_grads_for_grad_norm()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            param_norm = param.grad.data.norm(2)  # L2 范数
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared



def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    # Check the availability of apex
    assert multi_tensor_applier is not None and amp_C is not None, \
        "apex is not available, please install it from https://github.com/NVIDIA/apex"

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across tensor, pipeline and expert model-parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_and_expert_parallel_group())
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
    return norm_2.item() ** 0.5




def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def _initialize_tp_communicators():
    """ initializing the communicators with user buffers for high-performance tensor-model-parallel 
        communication overlap """

    try:
       import yaml
       from transformer_engine.pytorch import module as te_module

    except ImportError:
       raise RuntimeError("Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
             "'transformer_engine' packages") 

    args = get_args()

    if args.tp_comm_overlap_cfg is not None:
       with open(args.tp_comm_overlap_cfg,"r") as stream:    
          ub_cfgs = yaml.safe_load(stream)
    else:
       ub_cfgs = {}

    input_shape = [(args.seq_length * args.micro_batch_size) // args.context_parallel_size , args.hidden_size]

    #We create a MPI process group, which is needed to bootstrap the pipelined 
    #tensor-model-parallel communication overlap
    torch.distributed.new_group(backend='mpi')

    te_module.base.initialize_ub(shape = input_shape, tp_size = args.tensor_model_parallel_size, 
                                 use_fp8 = (args.fp8 is not None) , ub_cfgs = ub_cfgs,)


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def get_transformer_model_group(check_initialized=True):
    """Get the transformer_model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TRANSFORMER_MODEL_GROUP is not None
        ), 'tensor context parallel group is not initialized'
    # print("get transformer blocks: ", dist.get_rank(_TRANSFORMER_MODEL_GROUP))
    if dist.get_rank(_TRANSFORMER_MODEL_GROUP) == -1:
        return None
    
    return _TRANSFORMER_MODEL_GROUP

def get_batch(data_iterator):
    # get batches based on the TP_CP rank you are on
    batch = get_batch_on_this_tp_cp_rank_vast(data_iterator)
    # batch = get_batch_on_this_tp_rank_vast(data_iterator)
    return batch

def flow_loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)

    loss_wo_w = output_tensor[1].mean()
    averaged_loss_wo_w  = average_losses_across_data_parallel_group([loss_wo_w])
    loss_wo_w = loss_wo_w.unsqueeze(0)
    
    return loss, {"loss": averaged_loss[0], "loss_wo_w": averaged_loss_wo_w[0]}


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)

    loss_wo_w = output_tensor[1].mean()
    averaged_loss_wo_w  = average_losses_across_data_parallel_group([loss_wo_w])
    loss_wo_w = loss_wo_w.unsqueeze(0)

    loss_f1 = output_tensor[2].mean()
    averaged_loss_f1  = average_losses_across_data_parallel_group([loss_f1])
    loss_f1 = loss_f1.unsqueeze(0)

    return loss, {"loss": averaged_loss[0], "loss_wo_w": averaged_loss_wo_w[0], "loss_f1": averaged_loss_f1[0]}

def sr_loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)

    return loss, {"loss": averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    batch = get_batch(data_iterator)
    output_tensor_list = model(batch)
    # breakpoint()

    return output_tensor_list, loss_func

def deepspeed_forward_backward(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    forward_only,
    zero_optimizer,
):
    if isinstance(model, list):
        model = model[0]
    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext


    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = deepspeed_forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                is_first_microbatch=(i == 0),
            )
            if not forward_only:
                deepspeed_backward_step(zero_optimizer, input_tensor, output_tensor, output_tensor_grad, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = deepspeed_forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        is_first_microbatch=(num_microbatches == 1),
    )

    if not forward_only:
        deepspeed_backward_step(zero_optimizer, input_tensor, output_tensor, output_tensor_grad, config)

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store

def deepspeed_forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    is_first_microbatch=False,
):

    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        output_tensor, loss_func = forward_step_func(data_iterator, model)

    output_tensor = loss_func(output_tensor)
    loss, loss_reduced = output_tensor
    output_tensor = loss / num_microbatches
    forward_data_store.append(loss_reduced)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def deepspeed_backward_step(zero_optimizer, input_tensor, output_tensor, output_tensor_grad, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    zero_optimizer.backward(output_tensor[0], retain_graph=False)
    zero_optimizer.overlapping_partition_gradients_reduce_epilogue()
    # torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses

def unpack_tensors(packed_tensor, intervals):
    features = tuple([packed_tensor[intervals[i-1]:intervals[i]] for i in range(1, len(intervals))])
    return features



def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )

def get_batch_on_this_tp_cp_rank_vast(data_iterator):
    def _broadcast(item):
        if item is not None:
            import torch.distributed as dist
            rank = dist.get_rank()
            torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    transformer_dim = 4096
    img_dim = 20
    video_dim = 16
    image_scale = 8
    frame_scale = 4
    clip_dim=768

    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        
        sizes_info = {}
        type_info = {}
        batch=dict(data)
        dtype = torch.bfloat16

        from teletron.core.parallel_state import get_comm_pair
        comm_pair = get_comm_pair()

        tensors_info = torch.empty((16), device=torch.cuda.current_device(), dtype=torch.int32)
        req = dist.irecv(tensors_info, comm_pair.producer, tag=0)
        req.wait()
        # print(f"size info: {tensors_info}")

        args = get_args()
        if args.distributed_vae:
            transformer_embedding_size = tensors_info[0] * tensors_info[1] * tensors_info[2]
            clip_embedding_size = tensors_info[3] * tensors_info[4] * tensors_info[5]
            first_img_embedding_size = tensors_info[6] * tensors_info[7] * tensors_info[8] * tensors_info[9] * tensors_info[10]
            video_embedding_size = tensors_info[11] * tensors_info[12] * tensors_info[13] * tensors_info[14] * tensors_info[15]

            recv_tensor = torch.empty(( transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size ), device=torch.cuda.current_device(), dtype=torch.bfloat16)

            intervals = [0, 
                        transformer_embedding_size, 
                        transformer_embedding_size +clip_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size 
                        ]

            
            
            # print("tensor shape: ", prompt_ids[0].size(1))
            req = dist.irecv(recv_tensor, comm_pair.producer, tag = 0)
            req.wait()

            context, clip_feature, img_y, latents = unpack_tensors(recv_tensor, intervals)
            context = context.view(tensors_info[0] , tensors_info[1] , tensors_info[2])
            clip_feature = clip_feature.view(tensors_info[3] , tensors_info[4] , tensors_info[5])
            img_y = img_y.view(tensors_info[6] , tensors_info[7] , tensors_info[8] , tensors_info[9] , tensors_info[10])
            latents = latents.view(tensors_info[11] , tensors_info[12] , tensors_info[13] , tensors_info[14] , tensors_info[15])
        else:
            pass
        
        # image_size = data['images'].size()
        # # image_size[1] = 81


        

        # args = get_args()
        # if args.distributed_vae:
        #     token_length_size = 1
        #     transformer_embedding_size = max_length* transformer_dim
        #     clip_embedding_size = (max_length+2)*1280
        #     first_img_embedding_size = img_dim *( image_size[1]//frame_scale + 1)*(image_size[3]//image_scale)*(image_size[4]//image_scale)
        #     video_embedding_size = video_dim *( image_size[1]//frame_scale + 1)*(image_size[3]//image_scale)*(image_size[4]//image_scale)
            
        #     recv_tensor = torch.empty((token_length_size + transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size ), device=torch.cuda.current_device(), dtype=torch.bfloat16)

        #     intervals = [0, 
        #                 token_length_size, 
        #                 token_length_size + transformer_embedding_size, 
        #                 token_length_size + transformer_embedding_size +clip_embedding_size,
        #                 token_length_size + transformer_embedding_size +clip_embedding_size +first_img_embedding_size,
        #                 token_length_size + transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size 
        #                 ]

            
            
        #     # print("tensor shape: ", prompt_ids[0].size(1))
        #     req = dist.irecv(recv_tensor, comm_pair.producer, tag = 0)
        #     req.wait()

        #     token_length, context, clip_feature, img_y, latents = unpack_tensors(recv_tensor, intervals)
        #     context = context.view(1, token_length, transformer_dim)
        #     clip_feature = clip_feature.view(1, token_length+2, 1280)
        #     img_y = img_y.view(1, img_dim, ( image_size[1]//frame_scale + 1), image_size[3]//image_scale, image_size[4]//image_scale)
        #     latents = latents.view(1, video_dim, image_size[1]//frame_scale + 1, image_size[3]//image_scale, image_size[4]//image_scale)
        # else:
        #     # TODO: remove use fix data
        #     pass


        batch["context"] = context
        batch["clip_feature"] = clip_feature
        batch["img_emb_y"] = img_y
        batch["latents"] = latents
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.to(torch.cuda.current_device())
        for key, tensor in batch.items():
            sizes_info[key] = tensor.size() if tensor is not None and isinstance(tensor, torch.Tensor)  else len(tensor)
            type_info[key] = tensor.dtype if tensor is not None and isinstance(tensor, torch.Tensor) else type(tensor)

        # print("sizes_info: ", sizes_info)
        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info = torch.distributed.broadcast_object_list([type_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        for key, tensor in batch.items():
            if isinstance(tensor, list):
                torch.distributed.broadcast_object_list(tensor, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
            elif isinstance(tensor, torch.Tensor):
                _broadcast(tensor)
            else:
                raise NotImplementedError(f"Unsupported data type: {type(tensor)}")

    else:
        sizes_info_list = [None]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info_list =[None]
        torch.distributed.broadcast_object_list(type_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        
        batch = {}
        for key, value in sizes_info_list[0].items():
            dtype = type_info_list[0][key]
            
            if isinstance(dtype, torch.dtype):  # dtype 是 torch.float32 这种
                tensor = torch.empty(value, dtype=dtype, device=torch.cuda.current_device())
                _broadcast(tensor)
                batch[key] = tensor
            
            else:  # 表示这是个 list 类型对象
                tensor = [None]*value
                torch.distributed.broadcast_object_list(
                    tensor,
                    src=mpu.get_tensor_context_parallel_src_rank(),
                    group=mpu.get_tensor_context_parallel_group()
                )
                batch[key] = tensor
    
    return batch

def get_batch_on_this_tp_cp_rank_vast_dist(data_iterator):
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    transformer_dim = 4096
    img_dim = 20
    video_dim = 16
    image_scale = 8
    frame_scale = 4
    clip_dim=768

    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        
        sizes_info = {}
        type_info = {}
        batch=dict(data)
        dtype = torch.bfloat16

        from teletron.core.parallel_state import get_comm_pair, get_world_group
        comm_pair = get_comm_pair()
        tensors_info = torch.ones((16), device=torch.cuda.current_device(), dtype=torch.int32)
        req = dist.irecv(tensors_info, comm_pair.producer)
        req.wait()

        args = get_args()
        if args.distributed_vae:
            transformer_embedding_size = tensors_info[0] * tensors_info[1] * tensors_info[2]
            clip_embedding_size = tensors_info[3] * tensors_info[4] * tensors_info[5]
            first_img_embedding_size = tensors_info[6] * tensors_info[7] * tensors_info[8] * tensors_info[9] * tensors_info[10]
            video_embedding_size = tensors_info[11] * tensors_info[12] * tensors_info[13] * tensors_info[14] * tensors_info[15]
            noise_size = video_embedding_size

            recv_tensor = torch.empty(( transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size + noise_size ), device=torch.cuda.current_device(), dtype=torch.bfloat16)

            intervals = [0, 
                        transformer_embedding_size, 
                        transformer_embedding_size +clip_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size,
                        transformer_embedding_size +clip_embedding_size +first_img_embedding_size + video_embedding_size + noise_size
                        ]

            req = dist.irecv(recv_tensor, comm_pair.producer, tag = 0)
            req.wait()

            context, clip_feature, img_y, latents, noise = unpack_tensors(recv_tensor, intervals)
            context = context.view(tensors_info[0] , tensors_info[1] , tensors_info[2])
            clip_feature = clip_feature.view(tensors_info[3] , tensors_info[4] , tensors_info[5])
            img_y = img_y.view(tensors_info[6] , tensors_info[7] , tensors_info[8] , tensors_info[9] , tensors_info[10])
            latents = latents.view(tensors_info[11] , tensors_info[12] , tensors_info[13] , tensors_info[14] , tensors_info[15])
            noise = noise.view(tensors_info[11] , tensors_info[12] , tensors_info[13] , tensors_info[14] , tensors_info[15])
        else:
            pass

        batch={}
        batch["context"] = context
        batch["clip_feature"] = clip_feature
        batch["img_emb_y"] = img_y
        batch["latents"] = latents
        batch["noise"] = noise
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.to(torch.cuda.current_device())
        for key, tensor in batch.items():
            sizes_info[key] = tensor.size() if tensor is not None and isinstance(tensor, torch.Tensor)  else len(tensor)
            type_info[key] = tensor.dtype if tensor is not None and isinstance(tensor, torch.Tensor) else type(tensor)
        
        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info = torch.distributed.broadcast_object_list([type_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        for key, tensor in batch.items():
            if isinstance(tensor, list):
                torch.distributed.broadcast_object_list(tensor, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
            elif isinstance(tensor, torch.Tensor):
                _broadcast(tensor.contiguous())
            else:
                raise NotImplementedError(f"Unsupported data type: {type(tensor)}")
        
    else:
        sizes_info_list = [None]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        type_info_list =[None]
        torch.distributed.broadcast_object_list(type_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        
        batch = {}
        for key, value in sizes_info_list[0].items():
            dtype = type_info_list[0][key]
            
            if isinstance(dtype, torch.dtype):  # dtype 是 torch.float32 这种
                tensor = torch.empty(value, dtype=dtype, device=torch.cuda.current_device())
                _broadcast(tensor)
                batch[key] = tensor
            
            else:  # 表示这是个 list 类型对象
                tensor = [None]*value
                torch.distributed.broadcast_object_list(
                    tensor,
                    src=mpu.get_tensor_context_parallel_src_rank(),
                    group=mpu.get_tensor_context_parallel_group()
                )
                batch[key] = tensor 

    return batch

def get_batch_on_this_tp_cp_rank_vast_origin(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        
        batch = {
            'images':data["images"].cuda(non_blocking = True),
            'first_ref_image': data["first_ref_image"].cuda(non_blocking = True) if "first_ref_image" in data else None,
            'prompt_embeds': data["prompt_embeds"].cuda(non_blocking = True),
            'clip_text_embed': None if "clip_text_embed" not in data else data["clip_text_embed"].cuda(non_blocking = True)
        }

        # Step 1: 保存每部分的大小信息（只在 Rank 0 执行）
        sizes_info = {key: tensor.size() if tensor is not None else None for key, tensor in batch.items()}

        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        
        _broadcast(batch['images'])
        _broadcast(batch['first_ref_image'])
        _broadcast(batch['prompt_embeds'])
        _broadcast(batch['clip_text_embed'])

    else:
        sizes_info = None 
        sizes_info_list = [sizes_info]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        images=torch.empty(sizes_info_list[0]['images'], dtype=torch.float32, device = torch.cuda.current_device())
        first_ref_image=torch.empty(sizes_info_list[0]['first_ref_image'], dtype=torch.float32, device = torch.cuda.current_device())
        prompt_embeds=torch.empty(sizes_info_list[0]['prompt_embeds'], dtype=torch.float32, device = torch.cuda.current_device())
        clip_text_embed=torch.empty(sizes_info_list[0]['clip_text_embed'], dtype=torch.float32, device = torch.cuda.current_device())
        

        _broadcast(images)
        _broadcast(first_ref_image)
        _broadcast(prompt_embeds)
        _broadcast(clip_text_embed)

        batch = {
            'images':images,
            'first_ref_image': first_ref_image,
            'prompt_embeds': prompt_embeds,
            'clip_text_embed': clip_text_embed
        }

    return batch

def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    if config_class is None:
        config_class = TransformerConfig
    config_class = config_class 
    # breakpoint()

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['gradient_accumulation_fusion'] = False
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        kw_args['activation_func'] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    # breakpoint()

    # Return config.
    return config_class(**kw_args)

###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))

@jit_fuser
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def bias_dropout_add(x, bias, residual, prob, training):
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@jit_fuser
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)

def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        args.ffn_hidden_size // args.tensor_model_parallel_size,
        dtype=dtype,
        device="cuda",
    )
    input = torch.rand(
        (
            args.seq_length,
            args.micro_batch_size,
            args.ffn_hidden_size // args.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(
        residual
    )
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(
        [False, True], [True, True], [True, True]
    ):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = (
        args.num_attention_heads / args.tensor_model_parallel_size
    ) * args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16
        and seq_len <= 16384
        and seq_len % 4 == 0
        and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not (
        (args.fp16 or args.bf16)
        and custom_kernel_constraint
        and args.masked_softmax_fusion
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        fused_kernel_load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernel_load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
            )
            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )




def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    group.add_argument('--fp8-format', default=None,
                       choices=['e4m3', 'hybrid'],
                       help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',
                       dest='fp8')
    group.add_argument('--fp8-margin', type=int, default=0,
                       help='Scaling margin for fp8',
                       dest='fp8_margin')
    group.add_argument('--fp8-interval', type=int, default=1,
                       help='Scaling update interval for fp8',
                       dest='fp8_interval')
    group.add_argument('--fp8-amax-history-len', type=int, default=1,
                       help='Number of steps for which amax history is recorded per tensor',
                       dest='fp8_amax_history_len')
    group.add_argument('--fp8-amax-compute-algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history',
                       dest='fp8_amax_compute_algo')
    group.add_argument('--no-fp8-wgrad', action='store_false',
                       help='Execute wgrad in higher precision even for FP8 runs',
                       dest='fp8_wgrad')
    group.add_argument('--transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')

    return parser

def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')

    group.add_argument('--inference-batch-times-seqlen-threshold',
                       type=int, default=512,
                       help='During inference, if batch-size times '
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--max-tokens-to-oom',
                       type=int, default=12000,
                       help='Maximum number of tokens during inference'
                       'tokens here is # in prompt + # to generate'
                       'Allows us to throw an error before OOM crashes server')
    group.add_argument('--output-bert-embeddings', action='store_true',
                       help='Output Bert embeddings (via mean pooling) from '
                       'model, rather than its binary head output or entire '
                       'hidden batch.')
    group.add_argument('--bert-embedder-type', default="megatron",
                       choices=["megatron", "huggingface"],
                       help='Select either Megatron or Huggingface as the '
                       'Bert embedder.')

    return parser


def _add_retro_args(parser):
    group = parser.add_argument_group(title='retro')

    group.add_argument('--retro-project-dir', default=None,
                       help='Retro project directory, which contains the '
                       'preprocessed data for pretraining. This directory '
                       'is built during preprocessing (see '
                       'tools/retro/README.md), and contains subdirectories '
                       'for the chunk database and pretraining neighbors.')
    group.add_argument('--retro-add-retriever',
                       action='store_true', default=False,
                       help='Add a retriever to the transformer, for use in '
                       'pretraining a Retro model.')
    group.add_argument('--retro-cyclic-train-iters', type=int, default=None,
                       help='Set number of training iterations for cyclic '
                       'Retro training.')
    group.add_argument('--retro-encoder-layers', type=int, default=2,
                       help='Number of layers to use for the retrieval '
                       'encoder.')
    group.add_argument('--retro-encoder-hidden-dropout',
                       type=float, default=0.1, help='Hidden dropout for '
                       'retrieval encoder.')
    group.add_argument('--retro-encoder-attention-dropout',
                       type=float, default=0.1, help='Attention dropout for '
                       'retrieval encoder.')
    group.add_argument("--retro-num-neighbors", type=int, default=2,
                       help='Number of neighbors to retrieve during '
                       'pretraining.')
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2,
                       help='Number of chunks to retrieve from the retrieval '
                       'database.')
    group.add_argument("--retro-attention-gate", type=float, default=1,
                       help="Gated cross attention.")
    group.add_argument("--retro-no-verify-neighbor-count", action="store_false",
                       dest="retro_verify_neighbor_count",
                       help="Skip verifying that len(GPT dataset) == len(saved "
                       "neighbors).")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--encoder-num-layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder-num-layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--group-query-attention', action='store_true',
                          help='Use group-query attention.')
    group.add_argument('--num-query-groups', type=int, default=1)

    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--position-embedding-type', type=str, default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')
    group.add_argument('--use-rotary-position-embeddings', action='store_true',
                       help='Use rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')
    group.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%%')
    group.add_argument('--rotary-interleaved', action='store_true',
                          help='Use interleaved rotary embedding.')
    group.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=None,
                       help='Sequence length interpolation factor for rotary embeddings.')
    group.add_argument('--no-position-embedding',
                       action='store_false',
                       help='Disable position embedding. Deprecated: use --position-embedding-type',
                       dest='add_position_embedding')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Which normalization technique to use.')
    group.add_argument('--norm-epsilon', type=float, default=1e-5,
                       help='Epsilon for layer norm and RMS norm.')
    group.add_argument('--apply-layernorm-1p', action='store_true',
                       help='Adjust LayerNorm weights such that they are centered '
                       'around zero. This improves numerical stability.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--squared-relu', action='store_true',
                       help='Use squared relu activation instead of default gelu')
    group.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.'),
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')
    group.add_argument('--producer-log-level', type=int,
                       default=2, choices=range(1,3),
                       help='logging level to producer detail. '
                       '   1: DEBUG LEVEL '
                       '   2: INFO LEVEL')
    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--log-throughput', action='store_true',
                       help='If set, calculate and log throughput per GPU.')
    group.add_argument('--log-progress', action='store_true',
                       help='If set, log progress (in terms of number of processed tokens and '
                       'number of floating-point operations) to progress.txt file in checkpoint '
                       'directory.')
    group.add_argument('--timing-log-level', type=int,
                       default=0, choices=range(0,3),
                       help='Granularity level to measure and report timing. '
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--no-barrier-with-level-1-timing', action='store_false',
                       help='If not set, use barrier with level 1 time '
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.',
                       dest='barrier_with_L1_time')
    group.add_argument('--timing-log-option', type=str, default='minmax',
                       choices=['max', 'minmax', 'all'],
                       help='Options for logging timing:'
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--log-batch-size-to-tensorboard', action='store_true',
                       help='If set, write batch-size to tensorboard.')
    group.add_argument('--no-log-gradient-norm-to-tensorboard',
                       action='store_false',
                       help='Disable gradient norm logging to tensorboard.',
                       dest='log_gradient_norm_to_tensorboard')
    group.add_argument('--no-log-learnig-rate-to-tensorboard',
                       action='store_false',
                       help='Disable learning rate logging to tensorboard.',
                       dest='log_learning_rate_to_tensorboard')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log-memory-to-tensorboard',
                       action='store_true',
                       help='Enable memory logging to tensorboard.')
    group.add_argument('--log-world-size-to-tensorboard',
                       action='store_true',
                       help='Enable world size logging to tensorboard.')
    group.add_argument('--wandb-project', type=str, default='',
                       help='The wandb project name. Ignore wandb by default.')
    group.add_argument('--wandb-exp-name', type=str, default='',
                       help='The wandb experiment name.')
    group.add_argument('--wandb-save-dir', type=str, default='',
                       help='Path to save the wandb results locally.')
    group.add_argument('--enable-one-logger', action='store_true',
                       help='If set, use one_logger to track E2E metrics'
                       'Note that one_logger is an internal tool and not available externally. '
                       'For installation, please try command: `pip install '
                       '--index-url=https://sc-hw-artf.nvidia.com/api/pypi/hwinf-ml-pypi/simple'
                       ' one_logger` or go to https://gitlab-master.nvidia.com/hwinf-dcm/onelogger '
                       'for more details')
    group.add_argument('--one-logger-project', type=str, default='e2e-tracking',
                       help='The one-logger project name. Will ignore if '
                       '--enable-one-logger is not set')
    group.add_argument('--one-logger-entity', type=str, default='hwinf_dcm',
                       help='The one-logger username or team name. Will ignore if '
                       '--enable-one-logger is not set')
    group.add_argument('--one-logger-run-name', type=str, default=None,
                       help='The one-logger run name displayed. Will ignore if '
                       '--enable-one-logger is not set')
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--start-weight-decay', type=float,
                       help='Initial weight decay coefficient for L2 regularization.')
    group.add_argument('--end-weight-decay', type=float,
                       help='End of run weight decay coefficient for L2 regularization.')
    group.add_argument('--weight-decay-incr-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Weight decay increment function.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--config-path', type=str, default='config.teleai_i2v.config')
    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false',
                       help='Check for NaNs in loss and grad',
                       dest='check_for_nan_in_loss_and_grad')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=None,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--activation-offload', action="store_true", help='enable activation cpu offload on '
                        'transformer forward, requires using activation checkpointing (recompute)')
    group.add_argument('--no-clone-scatter-output-in-embedding', action='store_false',
                       help='If not set, clone the output of the scatter in embedding layer to GC original tensor.',
                       dest='clone_scatter_output_in_embedding')
    group.add_argument('--consumer-profile', action='store_true',
                       help='Enable transformer blocks torch profiling.')
    group.add_argument('--producer-profile', action='store_true',
                       help='Enable data producer torch profiling.')
    group.add_argument('--profile-path', type=str, default=None,
                       help='Directory to save torch profiling traces.')
    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,
                       help='Global step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Global step to stop profiling.')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')
    group.add_argument('--tp-comm-overlap', action='store_true', help='Enables the '
                       ' overlap of Tensor parallel communication and GEMM kernels.')
    group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,
                       help='Config file when tp_comm_overlap is enabled.')
    group.add_argument('--disable-tp-comm-overlap-ag', action='store_false', 
                       help=('Disables the All-Gather overlap with GEMM by '
                             'pipelining the GEMM and All-Gather.'),
                       dest='tp_comm_overlap_ag')
    group.add_argument('--disable-tp-comm-overlap-rs', action='store_false',
                       help=('Disables the Reduce-Scatter overlap with GEMM by '
                             'pipelining the GEMM and Reduce-Scatter.'),
                       dest='tp_comm_overlap_rs')
    group.add_argument('--disable-tp-comm-bulk-dgrad', action='store_false',
                       help='Disables the All-Gather overlap with bprop activation gradient GEMM.',
                       dest='tp_comm_bulk_dgrad')
    group.add_argument('--disable-tp-comm-bulk-wgrad', action='store_false',
                       help='Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.',
                       dest='tp_comm_bulk_wgrad')
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None,
                       help='If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism.')
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-swiglu-fusion', action='store_false',
                       help='Disable bias and swiglu fusion, the fusion is '
                       'available only when using megatron-core.',
                       dest='bias_swiglu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--no-rope-fusion', action='store_false',
                       help='Disable rope fusion, the fusion is available '
                       'only when using megatron-core.',
                       dest='apply_rope_fusion')
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--add-qkv-bias', action='store_true',
                       help='Enable bias only in the QKV linear layers',
                       dest='add_qkv_bias')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['common', 'single', 'cyclic', 'external'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--use-mcore-models', action='store_true',
                       help='Use the implementation from megatron core')
    group.add_argument('--manual-gc', action='store_true',
                       help='Disable the threshold-based default garbage '
                       'collector and trigger the garbage collection manually. '
                       'Manual garbage collection helps to align the timing of '
                       'the collection across ranks which mitigates the impact '
                       'of CPU-associated jitters. When the manual gc is enabled, '
                       'garbage collection is performed only at the start and the '
                       'end of the validation routine by default.')
    group.add_argument('--manual-gc-interval', type=int, default=0,
                       help='Training step interval to trigger manual garbage '
                       'collection. When the value is set to 0, garbage '
                       'collection is not triggered between training steps.')
    group.add_argument('--no-manual-gc-eval', action='store_false',
                       help='When using manual garbage collection, disable '
                       'garbage collection at the start and the end of each '
                       'evaluation run.', dest='manual_gc_eval')
    group.add_argument('--disable-tp-comm-split-ag', action='store_false',
                       help='Disables the All-Gather overlap with fprop GEMM.',
                       dest='tp_comm_split_ag')
    group.add_argument('--disable-tp-comm-split-rs', action='store_false',
                       help='Disables the Reduce-Scatter overlap with fprop GEMM.',
                       dest='tp_comm_split_rs')

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--data-parallel-random-init', action='store_true',
                       help='Enable random initialization of params '
                       'across data parallel ranks')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learning rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-init', type=float, default=0.0,
                       help='Initial value for learning rate warmup. The '
                       'scheduler starts warmup from this value.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minimum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    group.add_argument('--decoupled-lr', type=float, default=None,
                       help='Separate learning rate for the input and output layer')
    group.add_argument('--decoupled-min-lr', type=float, default=None,
                       help='Minimum value for learning rate for the input and output layer. The scheduler'
                       'clip values below this threshold')

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Directory containing a pretrained model checkpoint for finetuning.')
    group.add_argument('--ckpt-step', type=int, default=None,
                       help='Checkpoint step to load model from.')
    group.add_argument('--no-initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')
    group.add_argument('--use-checkpoint-args', action='store_true',
                       help='Override any command line arguments with arguments '
                       'from the checkpoint')
    group.add_argument('--exit-on-missing-checkpoint', action='store_true',
                       help="If '--load' is set, but checkpoint is not found "
                       "(e.g., path typo), then exit instead of random "
                       "initialization.")
    group.add_argument('--use-dist-ckpt', action='store_true',
                       help='Use distributed checkpoint format.')
    group.add_argument('--auto-detect-ckpt-format', action='store_true',
                       help='Determine if the checkpoint format is in legacy or distributed format.'
                            ' If False, expects distributed checkpoint iff args.use_dist_ckpt.'
                            ' Might slow down loading a bit (double rank0 ckpt load).')
    group.add_argument('--dist-ckpt-format', type=str, default='torch_dist',
                       choices=['zarr', 'torch_dist'],
                       help='Distributed checkpoint format to use.')
    group.add_argument('--ckpt-fully-parallel-save', action='store_true',
                       help='Apply full save parallelization across DP for'
                            ' distributed checkpoints. Depending on ckpt format'
                            ' might increase number of files in the checkpoint.')
    group.add_argument('--with-ema', action='store_true', help='save checkpoint with ema model')
    group.add_argument('--ema-decay', type=float, default=0.9999, help='decay of ema model')

    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scaling.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--apply-query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number. '
                       'Useful for fp16 training.')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32. '
                       'This flag is ignored unless '
                       '--no-query-key-layer-scaling is specified.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')
    group.add_argument('--no-delay-grad-reduce', action='store_false',
                       help='If not set, delay / synchronize grad reductions in all but first PP stage.',
                       dest='delay_grad_reduce')
    group.add_argument('--overlap-param-gather', action='store_true',
                       default=False, help='If set, overlap param all-gather in distributed optimizer.')
    group.add_argument('--delay-param-gather', action='store_true',
                       default=False, help='If set, delay / synchronize param all-gathers in all but first PP stage.')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='If not set, use scatter/gather to optimize communication of tensors in pipeline.',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    group.add_argument('--use-zero2', action='store_true',
                       help='Use DeepSpeed Zero2 distributed optimizer.')
    group.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')
    group.add_argument('--nccl-communicator-config-path', type=str, default=None,
                       help='Path to the yaml file with NCCL communicator '
                       'configurations. The number of min/max thread groups and thread '
                       'group cluster size of each communicator can be configured by '
                       'setting `min_ctas`, `max_ctas`, and `cga_cluster_size`.')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument("--test-mode", action="store_true", help='Run all real-time test alongside the experiment.')
    group.add_argument('--skip-train', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                       'optionally do evaluation for validation/test, and exit.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='Path to the test dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--data-cache-path', default=None,
                       help='Path to a directory to hold cached index files.')
    group.add_argument('--no-mmap-bin-files', action='store_false',
                       help='Disable mmap-ing of .bin files.',
                       dest='mmap_bin_files')
    group.add_argument('--mock-data', action='store_true',
                       help='Skip data loading and validation and opt for artificial '
                       'generation of mock data when an implementation is available.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser

def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")
    group.add_argument('--expert-model-parallel-size', type=int, default=1,
                       help='Degree of expert model parallelism.')
    group.add_argument('--num-experts', type=int, default=None,
                       help='Number of Experts in MoE (None means no MoE)')
    group.add_argument('--moe-router-load-balancing-type', type=str,
                       choices=['aux_loss', 'sinkhorn', "none"],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".')
    group.add_argument('--moe-router-topk', type=int, default=2,
                       help='Number of experts to route to for each token. The default is 2.')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).')
    group.add_argument('--moe-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    group.add_argument('--moe-z-loss-coeff', type=float, default=None,
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')
    group.add_argument('--moe-input-jitter-eps', type=float, default=None,
                       help='Add noise to the input tensor by applying jitter with a specified epsilon value.')
    group.add_argument('--moe-token-dropping', action='store_true',
                       help='This feature involves selectively dropping and padding tokens for each expert to achieve a specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note: Currently unsupported.')
    group.add_argument('--moe-token-dispatcher-type', type=str,
                       choices=['allgather', 'alltoall'],
                       default='allgather',
                       help='.')
    group.add_argument('--moe-per-layer-logging', action='store_true',
                       help='Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.')

    return parser

def _add_experimental_args(parser):
    group = parser.add_argument_group(title='experimental')

    group.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                       'that returns a spec to customize a model, transformer '
                       'block, or transformer layer, depending on the use case.'
                       'To use local spec specify local as the argument.'
                       'For more details, see the model class, '
                       '`transformer_block.py`, or `transformer_layer.py`')
    group.add_argument('--yaml-cfg', type=str, default=None,
                       help = 'Config file to add additional arguments')

    return parser
