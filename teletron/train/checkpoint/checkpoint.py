import os
import sys
import random
import torch
import numpy as np
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core import mpu, tensor_parallel, dist_checkpointing
from teletron.utils import (
    print_rank_0,
    get_args,
    update_num_microbatches,
)
from .utils import (
    _load_base_checkpoint,
    read_metadata,
    get_checkpoint_name,
    get_rng_state,
    get_checkpoint_tracker_filename,
    ensure_directory_exists,
    checkpoint_exists,
    get_distributed_optimizer_checkpoint_name,
)


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model

class CheckPointMixin:

    def save_checkpoint_and_time(self, iteration, model, optimizer, opt_param_scheduler,
                             num_floating_point_operations_so_far, ema_models):
        self.save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                        num_floating_point_operations_so_far, ema_models)

    def save_checkpoint(self, iteration, model, optimizer, opt_param_scheduler,
                num_floating_point_operations_so_far, ema_models):
        """Save a model checkpoint."""
        args = get_args()

        # Only rank zero of the data parallel writes to the disk.
        model = unwrap_model(model)

        ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
        print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
            iteration, args.save, ckpt_format))

        # Collect rng state across data parallel ranks.
        rng_state = get_rng_state(args.use_dist_ckpt)

        # Checkpoint name.
        checkpoint_name = get_checkpoint_name(args.save, iteration, return_base_dir=args.use_dist_ckpt)
        
        #save ema model
        if ema_models is not None:
            ema_config_list = []
            ema_state_dict_list = []
            for ema_model in ema_models:
                ema_state_dict_list.append(ema_model.state_dict())
                ema_config_list.append(ema_model.config)
            if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
                ema_checkpoint_name  = get_checkpoint_name(args.save, iteration, return_base_dir=args.use_dist_ckpt, ema=True)
                ema_config_name  = get_checkpoint_name(args.save, iteration, return_base_dir=args.use_dist_ckpt, ema_config=True)
                ensure_directory_exists(ema_checkpoint_name)
                torch.save(ema_state_dict_list, ema_checkpoint_name)
                torch.save(ema_config_list, ema_config_name)

        # Save distributed optimizer's custom parameter state.
        if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
            optim_checkpoint_name = \
                get_distributed_optimizer_checkpoint_name(checkpoint_name)
            ensure_directory_exists(optim_checkpoint_name)
            optimizer.save_parameter_state(optim_checkpoint_name)

        # Save Deepspeed Zero2 shared optimizer's parameter state.
        if args.use_zero2 and not args.no_save_optim and optimizer is not None:
            dp_rank = mpu.get_data_parallel_rank()
            cp_rank = mpu.get_context_parallel_rank()
            optim_checkpoint_name = \
                self.get_zero2_optimizer_checkpoint_name(checkpoint_name, dp_rank, cp_rank)
            ensure_directory_exists(optim_checkpoint_name)
            self.zero2_optimizer_save(optimizer, optim_checkpoint_name)

        # Collect args, model, RNG.
        if not torch.distributed.is_initialized() \
                or mpu.get_data_parallel_rank(with_context_parallel=True) == 0 \
                or args.use_dist_ckpt:

            optim_sd_kwargs = {}
            if args.use_dist_ckpt and args.use_distributed_optimizer:
                optim_sd_kwargs['sharding_type'] = ('fully_sharded_bucket_space'
                                                    if args.ckpt_fully_parallel_save
                                                    else 'dp_zero_gather_scatter')
                print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
            state_dict = CheckPointMixin.generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                            args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

            state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
            if args.use_dist_ckpt:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    ensure_directory_exists(checkpoint_name,
                                            check_parent=False)
                dist_checkpointing.save(state_dict, checkpoint_name, (args.dist_ckpt_format, 1))

            else:
                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)

        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \
                    .format(iteration, args.save))

        # And update the latest iteration
        if not torch.distributed.is_initialized() \
        or torch.distributed.get_rank() == 0:
            tracker_filename = get_checkpoint_tracker_filename(args.save)
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))

        # Wait so everyone is done (not necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, model, optimizer, opt_param_scheduler, load_arg='load', strict=True, ema_models=None):
        """Load a model checkpoint and return the iteration.
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` of the checkpoint match the names of
            parameters and buffers in model.
        """
        args = get_args()
        load_dir = getattr(args, load_arg)

        # Finetuning directories
        pretrained_dir = getattr(args,'pretrained_checkpoint', None)
        if pretrained_dir is not None and not checkpoint_exists(load_dir):
            print_rank_0(f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}')
            load_dir = pretrained_dir
            if not checkpoint_exists(load_dir):
                raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
            args.finetune = True


        model = unwrap_model(model)

        load_kwargs = {}
        is_dist_ckpt = False
        if args.auto_detect_ckpt_format or args.use_dist_ckpt:
            state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=True, exit_on_missing_checkpoint=args.exit_on_missing_checkpoint)
            is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
            if is_dist_ckpt:
                ckpt_tp_pp = (state_dict['args'].tensor_model_parallel_size, state_dict['args'].pipeline_model_parallel_size)
                run_tp_pp = (mpu.get_tensor_model_parallel_world_size(), mpu.get_pipeline_model_parallel_world_size())
                mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(ckpt_tp_pp, run_tp_pp)

                if ckpt_tp_pp == run_tp_pp and not getattr(state_dict['args'], 'no_save_rng', False):
                    rng_state = get_rng_state(True)  # we can load the rng state
                else:
                    rng_state = None
                    print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

                # TODO: add DistributedOptimizer support for differing TPxPP
                if ckpt_tp_pp != run_tp_pp and not release and not args.finetune and not args.no_load_optim and args.use_distributed_optimizer:
                    raise RuntimeError("{}: not supported for DistributedOptimizer".format(mismatch_msg))

                optim_sd_kwargs = dict(is_loading=True)
                if args.use_distributed_optimizer:
                    optim_sd_kwargs['sharding_type'] = ('fully_sharded_bucket_space'
                                                        if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                                                        else 'dp_zero_gather_scatter')
                load_kwargs['sharded_state_dict'] = CheckPointMixin.generate_state_dict(args, model, optimizer, opt_param_scheduler,
                                                                        rng_state, args.use_dist_ckpt, optim_sd_kwargs=optim_sd_kwargs)
                load_kwargs['exit_on_missing_checkpoint'] = args.exit_on_missing_checkpoint

        if args.use_zero2:
            state_dict, checkpoint_name, release = self._load_zero2_checkpoint(load_dir, **load_kwargs)
        else:
            state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=False, **load_kwargs)

        #load ema model
        if ema_models is not None:
            tracker_filename = get_checkpoint_tracker_filename(load_dir)
            iteration, release = read_metadata(tracker_filename)
            ema_checkpoint_name  = get_checkpoint_name(load_dir, iteration, return_base_dir=False, ema=True)
            ema_config_name  = get_checkpoint_name(load_dir, iteration, return_base_dir=False, ema_config=True)
            ema_models_state_dict = torch.load(ema_checkpoint_name, map_location='cpu', weights_only=False)
            ema_config_state_dict = torch.load(ema_config_name, map_location='cpu', weights_only=False)
            for ema_model, ema_state_dict, ema_config in zip(ema_models, ema_models_state_dict, ema_config_state_dict):
                ema_model.load_state_dict(ema_state_dict, device=torch.cuda.current_device(), dtype=torch.float32)
                ema_model.load_config(ema_config)
        
        # Checkpoint not loaded.
        if state_dict is None:
            # Iteration and num_floating_point_operations_so_far default to 0.
            return 0, 0, optimizer, opt_param_scheduler

        # Set checkpoint version.
        # set_checkpoint_version(state_dict.get('checkpoint_version', 0))

        # Set iteration.
        if args.finetune or release:
            iteration = 0
        else:
            try:
                iteration = state_dict['iteration']
            except KeyError:
                try:  # Backward compatible with older checkpoints
                    iteration = state_dict['total_iters']
                except KeyError:
                    print_rank_0('A metadata file exists but unable to load '
                                'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                    sys.exit()
        num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

        # Check arguments.
        assert args.consumed_train_samples == 0
        assert args.consumed_valid_samples == 0
        if 'args' in state_dict and not args.finetune and not release:
            checkpoint_args = state_dict['args']
            # check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args,
                                                'consumed_train_samples', 0)
            update_num_microbatches(consumed_samples=args.consumed_train_samples)
            args.consumed_valid_samples = getattr(checkpoint_args,
                                                'consumed_valid_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

        # Model.
        strict = False if args.retro_add_retriever else strict

        if args.lora:
            raise NotImplementedError('Lora not implement yet')
            from peft import get_peft_model, LoraConfig, TaskType
            from teletron.models.wan.light_pipeline import find_lora_target_modules
            base_model_path = args.lora_base_model_path
            target_modules = find_lora_target_modules(model[0], args.lora_target_modules)
            lora_config = LoraConfig(
                r=args.lora_rank,                             # LoRA rank
                lora_alpha=args.lora_alpha,                   # Scaling factor
                # target_modules=["q", "v","o","k"],  # Layer names to apply LoRA to
                target_modules=target_modules, # Layer names to apply LoRA to
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type= TaskType[args.lora_task_type]    # or SEQ_CLS, TOKEN_CLS etc.
            )
            if len(model) == 1:
                model[0].load_state_dict(torch.load(base_model_path, weights_only=False))
                model[0]=get_peft_model(model[0], lora_config)
                model[0].load_state_dict(state_dict['model'], strict=strict)
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model[i].load_state_dict(torch.load(base_model_path, weights_only=False), strict=strict)
                    model[i]=get_peft_model(model[i], lora_config)
                    model[i].load_state_dict(state_dict['model%d' % i], strict=strict)
        else:
            
            # Remove the 'transformer.' prefix from all keys
            # new_state_dict = {
            #     k.replace('transformer.', '', 1): v
            #     for k, v in state_dict['model'].items()
            #     if k.startswith('transformer.')
            # }

            if len(model) == 1:
                model[0].load_state_dict(state_dict['model'], strict=strict)
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

        # Fix up query/key/value matrix ordering if needed.
        # checkpoint_version = get_checkpoint_version()
        # print_rank_0(f' checkpoint version {checkpoint_version}')
        # fix_query_key_value_ordering(model, checkpoint_version)

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                # Load state dict.
                if optimizer is not None:
                    if args.use_zero2:
                        optimizer.load_state_dict(state_dict['optimizer'], load_from_fp32_weights=True)
                    else:
                        optimizer.load_state_dict(state_dict['optimizer'])

                # Load distributed optimizer's custom parameter state.
                # For distributed checkpoint it's already loaded in load_state_dict above
                if args.use_distributed_optimizer and not is_dist_ckpt:
                    tracker_filename = get_checkpoint_tracker_filename(load_dir)
                    iteration, release = read_metadata(tracker_filename)
                    model_checkpoint_name = \
                        get_checkpoint_name(load_dir, iteration, release)
                    optim_checkpoint_name = \
                        get_distributed_optimizer_checkpoint_name(
                            model_checkpoint_name)
                    optimizer.load_parameter_state(optim_checkpoint_name)

                # Load scheduler.
                if opt_param_scheduler is not None:
                    if 'lr_scheduler' in state_dict: # backward compatbility
                        opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                    else:
                        scheduler_dict = state_dict['opt_param_scheduler']
                        # num_steps = num_steps / ori_rank * new_rank
                        num_steps = scheduler_dict['num_steps'] / state_dict['args'].dit_world_size * args.dit_world_size
                        scheduler_dict['num_steps'] = num_steps
                        opt_param_scheduler.load_state_dict(scheduler_dict)
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}. '
                            'Specify --no-load-optim or --finetune to prevent '
                            'attempting to load the optimizer state, '
                            'exiting ...'.format(checkpoint_name))
                sys.exit()
        else:
            if (args.fp16 or args.bf16) and optimizer is not None:
                if args.use_zero2:
                    optimizer.refresh_fp32_params()
                else:
                    optimizer.reload_model_params()

        # rng states.
        if not release and not args.finetune and not args.no_load_rng:
            try:
                if 'rng_state' in state_dict:
                    # access rng_state for data parallel rank
                    if args.data_parallel_random_init:
                        rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                    else:
                        rng_state = state_dict['rng_state'][0]
                    random.setstate(rng_state['random_rng_state'])
                    np.random.set_state(rng_state['np_rng_state'])
                    torch.set_rng_state(rng_state['torch_rng_state'])
                    torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                    # Check for empty states array
                    if not rng_state['rng_tracker_states']:
                        raise KeyError
                    tensor_parallel.get_cuda_rng_tracker().set_states(
                        rng_state['rng_tracker_states'])
                else:  # backward compatability
                    random.setstate(state_dict['random_rng_state'])
                    np.random.set_state(state_dict['np_rng_state'])
                    torch.set_rng_state(state_dict['torch_rng_state'])
                    torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                    # Check for empty states array
                    if not state_dict['rng_tracker_states']:
                        raise KeyError
                    tensor_parallel.get_cuda_rng_tracker().set_states(
                        state_dict['rng_tracker_states'])
            except KeyError:
                print_rank_0('Unable to load rng state from checkpoint {}. '
                            'Specify --no-load-rng or --finetune to prevent '
                            'attempting to load the rng state, '
                            'exiting ...'.format(checkpoint_name))
                sys.exit()
                
        # TODO: need a more robust way to implement this function
        # for bucket sampler dataloader
        # args.last_micro_batch_access_index = state_dict["last_microbatch_size_index"]

        # Some utilities want to load a checkpoint without distributed being initialized
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                    f'[ t {mpu.get_tensor_model_parallel_rank()}, '
                    f'p {mpu.get_pipeline_model_parallel_rank()} ] '
                    f'at iteration {iteration}')

        return iteration, num_floating_point_operations_so_far, optimizer, opt_param_scheduler

    @staticmethod
    def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,sampler=None,
                        optim_sd_kwargs=None):
        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        if iteration is not None:
            state_dict['iteration'] = iteration
        # TODO:need a more robust way
        # save bucketSample last_microbatch_size_index
        # state_dict["last_microbatch_size_index"]=args.last_microbatch_size_index

        if len(model) == 1:
            state_dict['model'] = (model[0].sharded_state_dict()
                                if use_dist_ckpt else
                                model[0].state_dict_for_save_checkpoint())
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = (
                    model[i].sharded_state_dict()
                    if use_dist_ckpt else
                    model[i].state_dict_for_save_checkpoint())
        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                if not args.use_zero2:
                    state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
                                        if use_dist_ckpt else
                                        optimizer.state_dict())
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()
        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state
        return state_dict

    def get_zero2_optimizer_checkpoint_name(self, model_checkpoint_name, dp_rank, cp_rank):
        return os.path.join(os.path.dirname(model_checkpoint_name),
                            f"zero2_optim_dp{dp_rank}_cp{cp_rank}.pt")

    def zero2_optimizer_save(self, optimizer, optim_checkpoint_name):
        state_dict = optimizer.state_dict()
        torch.save(state_dict, optim_checkpoint_name)

    def load_zero2_optimizer(self, optimizer_state_dict_names, state_dict):
        optimizer_state_list = []
        for path in optimizer_state_dict_names:
            optim_dict = torch.load(path, map_location='cpu', weights_only=False)
            optimizer_state_list.append(optim_dict)
        state_dict['optimizer'] = optimizer_state_list

    def _load_zero2_checkpoint(self, load_dir, checkpoint_step = None):
        """ Load the base state_dict from the given directory

        If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
        """
        #import ipdb; ipdb.set_trace()
        # Read the tracker file and set the iteration.
        args = get_args()
        tracker_filename = get_checkpoint_tracker_filename(load_dir)

        # If no tracker file, return nothing
        if not os.path.isfile(tracker_filename):
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                            'random')
            return None, "", False

        # Otherwise, read the tracker file and either set the iteration or
        # mark it as a release checkpoint.
        if checkpoint_step is not None:
            iteration = checkpoint_step
            release = False
        else:
            iteration, release = read_metadata(tracker_filename)

        # Checkpoint.
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                                return_base_dir=True)
        is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)

        assert is_dist_ckpt == False, ("Zero2 optimizer not support dist ckpt format!")
        
        model_checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=False)
        if not release and not args.no_load_optim:
            optimizer_state_dict_names = get_checkpoint_name(load_dir, iteration, release, return_base_dir=False, use_zero2=True)

        dist_infix = "distributed " if is_dist_ckpt else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}')
        # Load the checkpoint.
        try:
            state_dict = torch.load(model_checkpoint_name, map_location='cpu', weights_only=False)
            if not release and not args.no_load_optim:
                self.load_zero2_optimizer(optimizer_state_dict_names, state_dict)
        except ModuleNotFoundError:
            # from megatron.legacy.fp16_deprecated import loss_scaler
            # For backward compatibility.
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler']
            sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler']
            sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
            state_dict = torch.load(model_checkpoint_name, map_location='cpu', weights_only=False)
            if not release and not args.no_load_optim:
                self.load_zero2_optimizer(optimizer_state_dict_names, state_dict)
            sys.modules.pop('fp16.loss_scaler', None)
            sys.modules.pop('megatron.fp16.loss_scaler', None)
            sys.modules.pop('megatron.model', None)
        except BaseException as e:
            print_rank_0('could not load the checkpoint')
            print_rank_0(e)
            sys.exit()

        return state_dict, checkpoint_name, release