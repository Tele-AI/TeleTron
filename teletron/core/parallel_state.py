# Copyright (c) 2025 TeleAI-infra and Nvidia Megatron-LM Team. All rights reserved.

from functools import wraps

import torch
import torch.distributed as dist
import megatron.core.parallel_state as ps

from typing import Optional
from datetime import timedelta
from functools import reduce
import operator
import os

_TENSOR_CONTEXT_PARALLEL_GROUP = None
_MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_CONTEXT_PARALLEL_RANK = None
# TP DP CP PP EP altogether, except dist-vae, etc.
_TRANSFORMER_MODEL_GROUP = None
_TRANSFORMER_THIS_MODEL_GROUP = None
# group that include all ranks
WORLD_GROUP = None 
# groups that include the first tp-cp ranks and the vae rank
_DATA_TRANSMIT_GROUP = []



from dataclasses import dataclass

@dataclass
class CommPair:
    producer: int
    consumer: int or list
    dp_rank: int
    dp_size: int

_DATA_PRODUCER_CONSUMER_GROUP=None

def apply_distributed_op_patches(models_num=1):
    if models_num > 1:
        torch_dist_barrier = torch.distributed.barrier 

        def dist_barrier_model_group(group=None, async_op=False, device_ids=None):
            if group is None:
                group = get_this_transformer_model_group()
            torch_dist_barrier(group=group, async_op=async_op, device_ids=device_ids)

        torch.distributed.barrier = dist_barrier_model_group

        torch_dist_get_rank = torch.distributed.get_rank 

        def get_rank_model_group(group=None):
            if group is None:
                group = get_this_transformer_model_group()
            return torch_dist_get_rank(group=group)

        torch.distributed.get_rank = get_rank_model_group

        all_reduce = torch.distributed.all_reduce

        def all_reduce_model_group(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
            if group is None:
                group = get_this_transformer_model_group()
            return all_reduce(tensor, op=op, group=group, async_op=async_op)
        
        torch.distributed.all_reduce = all_reduce_model_group

        all_gather_base = torch.distributed._all_gather_base

        def all_gather_base_model_group(output, input, group=None, async_op=False):
            if group is None:
                group = get_this_transformer_model_group()
            return all_gather_base(output, input, group=group, async_op=async_op)
        
        torch.distributed._all_gather_base = all_gather_base_model_group

        get_world_size = torch.distributed.get_world_size

        def get_world_size_model_group(group=None):
            if group is None:
                group = get_this_transformer_model_group()
            
            return get_world_size(group=group)
        
        torch.distributed.get_world_size = get_world_size_model_group

        broadcast = torch.distributed.broadcast

        def broadcast_model_group(tensor, src=None, group=None):
            if group is None:
                group = get_this_transformer_model_group()
            group_ranks = dist.get_process_group_ranks(group)
            while (src in group_ranks) is False:
                src += dist.get_world_size(group=get_this_transformer_model_group())
            return broadcast(tensor, src=src, group=group)
        
        torch.distributed.broadcast = broadcast_model_group

        broadcast_object_list = torch.distributed.broadcast_object_list

        def broadcast_object_list_model_group(object_list, src=None, group=None, device=None):
            if group is None:
                group = get_this_transformer_model_group()
            group_ranks = dist.get_process_group_ranks(group)
            while (src in group_ranks) is False:
                src += dist.get_world_size(group=get_this_transformer_model_group())
            return broadcast_object_list(object_list, src=src, group=group, device=device)

        torch.distributed.broadcast_object_list = broadcast_object_list_model_group
    else:
        torch_dist_barrier = torch.distributed.barrier 

        def dist_barrier_model_group(group=None, async_op=False, device_ids=None):
            if group is None:
                group = get_transformer_model_group()
            torch_dist_barrier(group=group, async_op=async_op, device_ids=device_ids)

        torch.distributed.barrier = dist_barrier_model_group

        all_reduce = torch.distributed.all_reduce

        def all_reduce_model_group(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
            if group is None:
                group = get_transformer_model_group()
            return all_reduce(tensor, op=op, group=group, async_op=async_op)
        
        torch.distributed.all_reduce = all_reduce_model_group

        all_gather_base = torch.distributed._all_gather_base

        def all_gather_base_model_group(output, input, group=None, async_op=False):
            if group is None:
                group = get_transformer_model_group()
            return all_gather_base(output, input, group=group, async_op=async_op)
        
        torch.distributed._all_gather_base = all_gather_base_model_group

        get_world_size = torch.distributed.get_world_size

        def get_world_size_model_group(group=None):
            if group is None:
                group = get_transformer_model_group()
            
            return get_world_size(group=group)
        
        torch.distributed.get_world_size = get_world_size_model_group

        broadcast = torch.distributed.broadcast

        def broadcast_model_group(tensor, src=None, group=None):
            if group is None:
                group = get_transformer_model_group()
            
            return broadcast(tensor, src=src, group=group)
        
        torch.distributed.broadcast = broadcast_model_group

def initialize_model_parallel_decorators(initialize_model_parallel):

    @wraps(initialize_model_parallel)
    def wrapper(tensor_model_parallel_size: int = 1,
                pipeline_model_parallel_size: int = 1,
                virtual_pipeline_model_parallel_size: Optional[int] = None,
                pipeline_model_parallel_split_rank: Optional[int] = None,
                use_sharp: bool = False,
                context_parallel_size: int = 1,
                expert_model_parallel_size: int = 1,
                nccl_communicator_config_path: Optional[str] = None,
                distributed_timeout_minutes: int = 30):
        
        global WORLD_GROUP
        WORLD_GROUP = torch.distributed.new_group(
            range(0, torch.distributed.get_world_size())
        )
        from teletron.utils import get_args
        margs = get_args()
        if margs.distributed_vae:
            
            extra_model_parallel_world_size = margs.distributed_vae_world_size
            total_world_size = torch.distributed.get_world_size()
            models_num = margs.consumer_models_num
            model_world_size = (total_world_size - extra_model_parallel_world_size)
        else: 
            models_num=1
            model_world_size = torch.distributed.get_world_size()
    

        ranks = range(0, model_world_size)
        base_process_group = torch.distributed.new_group(
            ranks
        )
        global _TRANSFORMER_MODEL_GROUP
        _TRANSFORMER_MODEL_GROUP = base_process_group

        global _TRANSFORMER_THIS_MODEL_GROUP
        per_model_world_size = model_world_size // models_num
        if models_num > 1:
            
            if torch.distributed.get_rank() < model_world_size:
                for k in range(models_num):
                    this_start_rank = k * per_model_world_size
                    ranks = range(this_start_rank, this_start_rank + per_model_world_size)
                    base_process_group = torch.distributed.new_group(
                        ranks
                    )
                    if torch.distributed.get_rank() in ranks:
                        _TRANSFORMER_THIS_MODEL_GROUP = base_process_group
        
        # build DATA_TRANSMIT_GROUP
        global _DATA_TRANSMIT_GROUP

        local_rank = torch.distributed.get_rank()
        
        tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
        num_tensor_and_context_groups: int = per_model_world_size  // tensor_and_context_group_size


        if get_transformer_model_group() is not None:
            print("**********start init MP**********************************")
            initialize_model_parallel_base(
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank,
                use_sharp,
                context_parallel_size,
                expert_model_parallel_size,
                nccl_communicator_config_path,
                distributed_timeout_minutes,
                _TRANSFORMER_THIS_MODEL_GROUP if models_num>1 else _TRANSFORMER_MODEL_GROUP
            )
            
            if margs.distributed_vae:
                initialize_comm_pair( tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
        else:
            print("vae data transmit group", _DATA_TRANSMIT_GROUP, flush=True)
            print("**********start init VAE**********************************")
            if margs.distributed_vae:
                initialize_comm_pair( tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
            return wrapper

        apply_distributed_op_patches(models_num)

    return wrapper

def initialize_comm_pair( tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
    from teletron.utils import get_args
    args = get_args()
    models_num = args.consumer_models_num
    world_size = args.dit_world_size                    
    model_world_size =  args.dit_world_size // models_num
    producer_size = args.distributed_vae_world_size

    # pp_start_rank = 0
    pp_size = model_world_size // pipeline_model_parallel_size

    global _DATA_PRODUCER_CONSUMER_GROUP

    local_rank = torch.distributed.get_rank()
    
    tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
    num_tensor_and_context_groups: int = pp_size // tensor_and_context_group_size
    assert num_tensor_and_context_groups % producer_size == 0 and num_tensor_and_context_groups//producer_size >=1
    if get_transformer_model_group() is not None:
        # consumer ranks
        for i in range(num_tensor_and_context_groups):
            start_rank = local_rank // model_world_size * model_world_size
            start_rank = i * tensor_and_context_group_size + start_rank
            if start_rank == local_rank:
                _DATA_PRODUCER_CONSUMER_GROUP = CommPair(
                    i%producer_size + world_size, local_rank, i, num_tensor_and_context_groups)
    else:
        for i in range(num_tensor_and_context_groups * models_num):
            if _DATA_PRODUCER_CONSUMER_GROUP is None:
                _DATA_PRODUCER_CONSUMER_GROUP = []
            start_rank = i * tensor_and_context_group_size
            if i%producer_size == local_rank - world_size:
                _DATA_PRODUCER_CONSUMER_GROUP.append(
                    CommPair(local_rank, start_rank, i%num_tensor_and_context_groups,num_tensor_and_context_groups)
                )

def get_comm_pair():
    return _DATA_PRODUCER_CONSUMER_GROUP

def get_world_group():
    return WORLD_GROUP

def initialize_model_parallel_base(tensor_model_parallel_size: int = 1,
                pipeline_model_parallel_size: int = 1,
                virtual_pipeline_model_parallel_size: Optional[int] = None,
                pipeline_model_parallel_split_rank: Optional[int] = None,
                use_sharp: bool = False,
                context_parallel_size: int = 1,
                expert_model_parallel_size: int = 1,
                nccl_communicator_config_path: Optional[str] = None,
                distributed_timeout_minutes: int = 30,
                base_process_group=None):   
    
    assert torch.distributed.is_initialized()
    if base_process_group == -100:
        from teletron.utils import get_args
        margs = get_args()
        extra_model_parallel_world_size = margs.distributed_vae_world_size
        total_world_size = torch.distributed.get_world_size()
        world_size = total_world_size - extra_model_parallel_world_size
    else:
        world_size = torch.distributed.get_world_size(base_process_group)


    print(f"base_process_group {base_process_group}, {world_size}\n"*5, flush=True)

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    from teletron.utils import get_args
    args = get_args()
    models_num = args.consumer_models_num

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        # global ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        # global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    all_data_parallel_group_ranks_with_cp = []


    for k in range(models_num):
        this_start_rank = k* world_size
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups + this_start_rank
            end_rank = (i + 1) * num_pipeline_model_parallel_groups + this_start_rank
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=ps.get_nccl_options('dp', nccl_comm_cfgs)
                )

                group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")

                if rank in ranks:
                    ps._DATA_PARALLEL_GROUP = group
                    ps._DATA_PARALLEL_GROUP_GLOO = group_gloo
                    ps._DATA_PARALLEL_GLOBAL_RANKS = ranks

            
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                group_with_cp = torch.distributed.new_group(
                    ranks_with_cp, timeout=timeout, pg_options=ps.get_nccl_options('dp_cp', nccl_comm_cfgs)
                )
                group_with_cp_gloo = torch.distributed.new_group(
                    ranks_with_cp, timeout=timeout, backend="gloo"
                )
                
                if rank in ranks_with_cp:
                    ps._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                    ps._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                    ps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    

    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=ps.get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    for t in range(models_num):
        this_start_rank = t* world_size
        for i in range(pipeline_model_parallel_size):
            for j in range(data_parallel_size):
                start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size + this_start_rank
                )
                end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size + this_start_rank
                )
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    group = torch.distributed.new_group(
                        ranks, timeout=timeout, pg_options=ps.get_nccl_options('cp', nccl_comm_cfgs)
                    )
                    if rank in ranks:
                        ps._CONTEXT_PARALLEL_GROUP = group
                        ps._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
        

    # Build the model-parallel groups.
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i]
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    
    for i in range(num_tensor_model_parallel_groups):
        start_rank = torch.distributed.get_rank() // world_size * world_size
        ranks = range(i * tensor_model_parallel_size + start_rank, 
                                    (i + 1) * tensor_model_parallel_size + start_rank)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._TENSOR_MODEL_PARALLEL_GROUP = group

    

    for k in range(models_num):
        this_start_rank = k* world_size
        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        for i in range(num_pipeline_model_parallel_groups):
            # start_rank = torch.distributed.get_rank() // torch.cuda.device_count() * torch.cuda.device_count()
            ranks = range(i+this_start_rank, world_size + this_start_rank, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('pp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._PIPELINE_MODEL_PARALLEL_GROUP = group
                ps._PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(
                embedding_ranks, timeout=timeout, pg_options=ps.get_nccl_options('embd', nccl_comm_cfgs)
            )
            if rank in embedding_ranks:
                ps._EMBEDDING_GROUP = group
            if rank in ranks:
                ps._EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(
                position_embedding_ranks,
                timeout=timeout,
                pg_options=ps.get_nccl_options('embd', nccl_comm_cfgs),
            )
            if rank in position_embedding_ranks:
                ps._POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                ps._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        # Build the tensor + data parallel groups.
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        for i in range(num_tensor_and_data_groups_with_cp):
            # this_start_rank = torch.distributed.get_rank() // torch.cuda.device_count() * torch.cuda.device_count()
            start_rank = i * tensor_and_data_group_size_with_cp + this_start_rank
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
            # this_start_rank = torch.distributed.get_rank() // torch.cuda.device_count() * torch.cuda.device_count()
            
            for j in range(context_parallel_size):
                ranks = []
                for k in range(data_parallel_size):
                    start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * tensor_model_parallel_size
                        + k * tensor_model_parallel_size * context_parallel_size + this_start_rank
                    )
                    end_rank = start_rank + tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_dp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    ps._TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + context parallel groups
    global _TENSOR_CONTEXT_PARALLEL_GROUP
    assert (
    _TENSOR_CONTEXT_PARALLEL_GROUP is None
    ), 'Tensor + context parallel group is already initialized'
    tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
    num_tensor_and_context_groups: int = world_size // tensor_and_context_group_size
    print(f"world_size: {world_size}, {tensor_and_context_group_size}")
    for k in range(models_num):
        this_start_rank = k* world_size
        for i in range(num_tensor_and_context_groups):
            start_rank = i * tensor_and_context_group_size + this_start_rank
            end_rank = start_rank + tensor_and_context_group_size
            ranks = range(start_rank, end_rank)
            
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_CONTEXT_PARALLEL_GROUP = group

        # Build the tensor + expert parallel groups
        tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size // expert_model_parallel_size
        for i in range(num_tensor_and_data_groups):
            for j in range(num_expert_groups):
                # TPxEP Group
                start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size + this_start_rank
                end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size + this_start_rank
                ranks = range(start_rank, end_rank)
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    ps._TENSOR_AND_EXPERT_PARALLEL_GROUP = group
                for k in range(tensor_model_parallel_size):
                    ranks = range(
                        start_rank + k, end_rank, tensor_model_parallel_size
                    )
                    group = torch.distributed.new_group(
                        ranks, pg_options=ps.get_nccl_options('exp', nccl_comm_cfgs)
                    )
                    if rank in ranks:
                        ps._EXPERT_MODEL_PARALLEL_GROUP = group

        for i in range(num_tensor_and_data_groups):
            start_rank = i * tensor_and_data_group_size + this_start_rank
            end_rank = (i + 1) * tensor_and_data_group_size + this_start_rank
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=ps.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    ps._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo
        
    

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    if ps._GLOBAL_MEMORY_BUFFER is None:
        ps._set_global_memory_buffer()


def initialize_model_parallel_extra(tensor_model_parallel_size: int = 1,
                pipeline_model_parallel_size: int = 1,
                virtual_pipeline_model_parallel_size: Optional[int] = None,
                pipeline_model_parallel_split_rank: Optional[int] = None,
                use_sharp: bool = False,
                context_parallel_size: int = 1,
                expert_model_parallel_size: int = 1,
                nccl_communicator_config_path: Optional[str] = None,
                distributed_timeout_minutes: int = 30,
                extra_model_parallel_world_size: int = 0,
                initialize_extra_model_parallel = None):
    if extra_model_parallel_world_size == 0:
        pass

def initialize_model_parallel_decorator_v2(initialize_model_parallel):
    def wrapper(
                *args, **kwargs
    ):
        initialize_model_parallel_extra()


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

def get_this_transformer_model_group(check_initialized=True):
    """Get the transformer_model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TRANSFORMER_THIS_MODEL_GROUP is not None
        ), 'tensor context parallel group is not initialized'
    # print("get transformer blocks: ", dist.get_rank(_TRANSFORMER_MODEL_GROUP))
    if dist.get_rank(_TRANSFORMER_THIS_MODEL_GROUP) == -1:
        return None
    
    return _TRANSFORMER_THIS_MODEL_GROUP


def get_tensor_context_parallel_group(check_initialized=True):
    """Get the tensor context parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_CONTEXT_PARALLEL_GROUP is not None
        ), 'tensor context parallel group is not initialized'
    return _TENSOR_CONTEXT_PARALLEL_GROUP

def get_tensor_context_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_context_parallel_group())

def get_tensor_context_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    if _MPU_TENSOR_CONTEXT_PARALLEL_RANK is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_context_parallel_group())

def get_tensor_context_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_context_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
    
def destroy_model_parallel_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()

        global _TENSOR_CONTEXT_PARALLEL_GROUP
        _TENSOR_CONTEXT_PARALLEL_GROUP = None
        global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
        _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
        global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
        _MPU_TENSOR_CONTEXT_PARALLEL_RANK= None
