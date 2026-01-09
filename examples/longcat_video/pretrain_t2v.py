import torch
import torch.distributed as dist
from megatron.core import mpu
from teletron.train import Trainer, parse_args
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import flow_loss_func
from teletron.utils import get_args
from teletron.utils import get_timers
import os


def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    return parser


def forward_step(data_iterator, model, time_step=None):
    args = get_args()
    flow_scheduler = FlowMatchScheduler(shift=1, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)

    timers = get_timers()
    timers.start_timer('get-data-time')
    batch = next(data_iterator)
    timers.stop_timer('get-data-time')

    latents = batch["latents"]
    noise = torch.randn_like(latents)
    timestep_range = [0, flow_scheduler.num_train_timesteps]

    timestep_id = torch.randint(timestep_range[0], timestep_range[1], (1,))
    timestep_dtype = torch.float32
    if getattr(args, "bf16", False):
        timestep_dtype = torch.bfloat16
    elif getattr(args, "fp16", False):
        timestep_dtype = torch.float16
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=timestep_dtype, device=torch.cuda.current_device()
    )

    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    if time_step is not None:
        timestep = torch.tensor([time_step], dtype=timestep_dtype, device=torch.cuda.current_device())

    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)

    training_target = flow_scheduler.training_target(latents, noise, timestep)
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    loss_weight = flow_scheduler.training_weight(timestep)

    context = batch["context"]
    if context.dim() == 3:
        context = context.unsqueeze(1)
        
    clip_feature = batch.get("img_clip_feature", None)
    img_emb_y = batch.get("img_emb_y", None)

    if img_emb_y is not None:
        assert img_emb_y.dim() == 5 and noisy_latents.dim() == 5
        assert img_emb_y.shape[:1] == noisy_latents.shape[:1]
        if img_emb_y.shape[1] != noisy_latents.shape[1]:
            img_emb_y = img_emb_y[:, -noisy_latents.shape[1]:, :, :, :]
        noisy_latents = torch.cat([img_emb_y, noisy_latents], dim=2)
        num_cond_latents = img_emb_y.shape[2]
    else:
        num_cond_latents = 0
    
    import inspect
    forward_params = inspect.signature(model.forward).parameters
    call_kwargs = dict(
        hidden_states=noisy_latents,
        timestep=timestep,
        encoder_hidden_states=context,
        num_cond_latents=num_cond_latents,
    )
    if "clip_feature" in forward_params and clip_feature is not None:
        call_kwargs["clip_feature"] = clip_feature
    
    pred = model(**call_kwargs)
    if num_cond_latents > 0:
        pred = pred[:, :, num_cond_latents:, :, :]
    
    loss_wo_w = torch.nn.functional.mse_loss(
        pred.float(), training_target.float()
    )
    loss = loss_wo_w * loss_weight
    return [loss, loss_wo_w], flow_loss_func


if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)

    from teletron.datasets.samplers import default_sampler
    _original_sampler_init = default_sampler.DefaultSampler.__init__

    def _patched_sampler_init(self, dataset, consumed_samples, micro_batch_size,
                              data_parallel_rank, data_parallel_size, global_batch_size, *args, **kwargs):
        # Check dataset size
        if hasattr(dataset, "__len__") and len(dataset) == 0:
            print(f"[Rank {torch.distributed.get_rank()}] CRITICAL WARNING: Dataset length is 0!")
        
        try:
            _original_sampler_init(self, dataset, consumed_samples, micro_batch_size,
                                   data_parallel_rank, data_parallel_size, global_batch_size, *args, **kwargs)
        except ZeroDivisionError:
            print(f"[Rank {torch.distributed.get_rank()}] ZeroDivisionError in DefaultSampler detected.")
            print(f"Dataset len: {len(dataset)}")
            print(f"Micro batch size: {micro_batch_size}")
            print(f"Data parallel size: {data_parallel_size}")
            # Try to calculate what happened
            total_samples = len(dataset)
            drop_last = kwargs.get('drop_last', True)
            if drop_last:
                num_samples = total_samples // micro_batch_size // data_parallel_size
            else:
                import math
                num_samples = math.ceil(math.ceil(total_samples / micro_batch_size) / data_parallel_size)
            print(f"Calculated num_samples: {num_samples}")
            print("Raising error again...")
            raise

    default_sampler.DefaultSampler.__init__ = _patched_sampler_init

    # -------------------------------------------------------------------------
    # PATCH: Support VAE at Rank 0, 1 and Train at Rank 2..9 (VAE-First Topology)
    # -------------------------------------------------------------------------
    import teletron.core.parallel_state as ps
    from functools import wraps
    from typing import Optional

    def patch_parallel_state_for_vae_first():
        """
        Patch parallel_state to support VAE at Rank 0, 1 and Train at Rank 2..9.
        """
        if not dist.is_initialized():
             return

        rank = dist.get_rank()
        print(f"[Patch] Applying VAE-First topology patch for Rank {rank}")
        
        # 1. Patch initialize_comm_pair
        def new_initialize_comm_pair(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
            args = get_args()
            models_num = args.consumer_models_num
            # args.dit_world_size is model_world_size (8) in this config
            world_size = args.dit_world_size 
            model_world_size = args.dit_world_size // models_num
            producer_size = args.distributed_vae_world_size
            
            pp_size = model_world_size // pipeline_model_parallel_size
            
            local_rank = torch.distributed.get_rank()
            
            tensor_and_context_group_size = tensor_model_parallel_size * context_parallel_size
            num_tensor_and_context_groups = pp_size // tensor_and_context_group_size
            
            # Determine if we are Consumer (Train) or Producer (VAE) based on rank
            # Train ranks: producer_size to producer_size + model_world_size - 1 (e.g., 2..9)
            # VAE ranks: 0 to producer_size - 1 (e.g., 0..1)
            
            is_train_node = (local_rank >= producer_size) and (local_rank < producer_size + model_world_size)
            
            if is_train_node:
                # We are Consumer (Train)
                for i in range(num_tensor_and_context_groups):
                    # Offset for train ranks
                    offset = producer_size
                    current_group_start_rank = i * tensor_and_context_group_size + offset
                    
                    # Check if local_rank falls in this group
                    if local_rank >= current_group_start_rank and local_rank < current_group_start_rank + tensor_and_context_group_size:
                         ps._DATA_PRODUCER_CONSUMER_GROUP = ps.CommPair(
                            i % producer_size, # Producer is 0, 1, ...
                            local_rank, 
                            i, 
                            num_tensor_and_context_groups
                         )
                         print(f"[Patch] Rank {local_rank} assigned as Consumer linked to Producer {i % producer_size}")
    
            else:
                # We are Producer (VAE)
                for i in range(num_tensor_and_context_groups * models_num):
                    if ps._DATA_PRODUCER_CONSUMER_GROUP is None:
                        ps._DATA_PRODUCER_CONSUMER_GROUP = []
                    
                    # Target start rank (Train)
                    target_start_rank = i * tensor_and_context_group_size + producer_size
                    
                    # Check if this group maps to us
                    if i % producer_size == local_rank:
                        ps._DATA_PRODUCER_CONSUMER_GROUP.append(
                            ps.CommPair(
                                local_rank, 
                                target_start_rank, # Base rank of consumer group
                                i % num_tensor_and_context_groups,
                                num_tensor_and_context_groups
                            )
                        )
                        print(f"[Patch] Rank {local_rank} assigned as Producer linked to Consumer Group starting at {target_start_rank}")
        
        ps.initialize_comm_pair = new_initialize_comm_pair

        # 2. Patch initialize_model_parallel_decorators
        # We need to patch the decorator generator itself or the result of it?
        # ps.initialize_model_parallel_decorators is a function that returns a decorator.
        # It is used in megatron_adaptor.py: 
        # megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorators(megatron.core.parallel_state.initialize_model_parallel)
        
        # Since megatron_adaptor might have already run (imported via teletron.train -> ...), 
        # we might be too late if we just patch ps.initialize_model_parallel_decorators.
        # BUT, pretrain_t2v.py imports Trainer, which imports teletron.train.
        # Let's check if megatron_adaptor is imported.
        # If it is already imported, megatron.core.parallel_state.initialize_model_parallel is ALREADY decorated.
        # So we need to re-decorate it or patch the wrapper.
        
        # However, looking at the code structure, the wrapper function is defined inside initialize_model_parallel_decorators.
        # We can't easily patch the inner wrapper.
        # But we can overwrite megatron.core.parallel_state.initialize_model_parallel AGAIN with our new wrapper.
        
        import megatron.core.parallel_state as mpu_ps
        
        def new_initialize_model_parallel_decorators(initialize_model_parallel):
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
            
                # Initialize WORLD_GROUP
                ps.WORLD_GROUP = torch.distributed.new_group(
                    range(0, torch.distributed.get_world_size())
                )
            
                from teletron.utils import get_args
                margs = get_args()
            
                if margs.distributed_vae:
                    extra_model_parallel_world_size = margs.distributed_vae_world_size
                    total_world_size = torch.distributed.get_world_size()
                    models_num = margs.consumer_models_num
                    model_world_size = (total_world_size - extra_model_parallel_world_size)
            
                    # PATCH: Use ranks [vae_size, total] for Transformer Group
                    start_rank = extra_model_parallel_world_size
                    ranks = range(start_rank, total_world_size)
                    print(f"[Patch] Initializing Transformer Group with ranks: {list(ranks)}")
                else:
                    models_num = 1
                    model_world_size = torch.distributed.get_world_size()
                    ranks = range(0, model_world_size)

                base_process_group = torch.distributed.new_group(ranks)
                ps._TRANSFORMER_MODEL_GROUP = base_process_group
            
                # Initialize Data Transmit Group (empty for now)
                ps._DATA_TRANSMIT_GROUP = []
            
                # Call initialize_model_parallel_base if we are in Transformer Group
                if ps.get_transformer_model_group() is not None:
                    print("**********start init MP (Patched)**********************************")
                    ps.initialize_model_parallel_base(
                        tensor_model_parallel_size,
                        pipeline_model_parallel_size,
                        virtual_pipeline_model_parallel_size,
                        pipeline_model_parallel_split_rank,
                        use_sharp,
                        context_parallel_size,
                        expert_model_parallel_size,
                        nccl_communicator_config_path,
                        distributed_timeout_minutes,
                        ps._TRANSFORMER_MODEL_GROUP # Assuming models_num=1
                    )
            
                    if margs.distributed_vae:
                        ps.initialize_comm_pair(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
                else:
                    print("**********start init VAE (Patched)**********************************")
                    if margs.distributed_vae:
                        ps.initialize_comm_pair(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
                    # For VAE nodes, we don't call the original initialize_model_parallel which sets up tensor/pipeline groups
                    return 
            
                ps.apply_distributed_op_patches(models_num)

            return wrapper

        # Apply the new decorator to the original (undecorated) initialize_model_parallel if possible?
        # Or just overwrite mpu_ps.initialize_model_parallel with our new wrapper.
        # But we need the ORIGINAL original.
        # Since we don't have it easily (it's already wrapped), let's assume we can just replace the implementation.
        # Actually, mpu_ps.initialize_model_parallel is the wrapped one.
        # We can just assign our new wrapper to it directly? No, it expects arguments.
        # We need to construct the wrapper.
        
        # Let's try to get the original function from the wrapper if possible, or just define a dummy original since we re-implement most logic.
        # But initialize_model_parallel_base calls mpu stuff.
        
        # Simpler approach: overwite mpu_ps.initialize_model_parallel with our `wrapper`.
        # We need to bind it.
        
        mpu_ps.initialize_model_parallel = new_initialize_model_parallel_decorators(lambda *args, **kwargs: None) 
        # Passing dummy lambda because our wrapper calls initialize_model_parallel_base directly and doesn't call the original function for VAE nodes.
        # For Transformer nodes, it calls initialize_model_parallel_base.
        # Wait, initialize_model_parallel_base is in teletron.core.parallel_state.
        # Does it call mpu.initialize_model_parallel?
        # No, it sets up groups manually.
        
        print("[Patch] Applied VAE-First topology patch to mpu.initialize_model_parallel")

    # Apply the patch
    rank = 0
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except:
        pass
    
    # We need to apply this BEFORE Trainer initializes MPU.
    # Trainer init calls mpu.initialize_model_parallel via arguments or internally?
    # Trainer.__init__ -> setup_parallel_state -> mpu.initialize_model_parallel
    
    patch_parallel_state_for_vae_first()
    
    print(f"[startup] rank={rank} config_path={getattr(args, 'config_path', None)}", flush=True)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
