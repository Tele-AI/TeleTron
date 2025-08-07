# Copyright (c) 2025 TeleAI-infra Team and Nvidia Megatron-LM Team. All rights reserved.

import torch
import torch.distributed as dist
import dataclasses
import time
import sys
import gc
import os
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.module import Float16Module
from megatron.core.enums import ModelType
from megatron.core.distributed import finalize_model_grads
from megatron.core import mpu, tensor_parallel
from megatron.core.optimizer import  OptimizerConfig
import deepspeed

from teletron.core.distributed import DistributedDataParallel as DDP
from teletron.utils import (
    print_rank_0,
    print_datetime,
    get_model_config,
    print_rank_last,
    is_last_rank,
    num_floating_point_operations,
    validate_args,
    set_args,
    get_args,
    update_num_microbatches,
    get_num_microbatches,
)
from teletron.train.utils import (
    _initialize_distributed,
    _compile_dependencies,
    set_jit_fusion_options,
    core_transformer_config_from_args,
    forward_step,
    deepspeed_forward_backward,
    _set_random_seed,
    _initialize_tp_communicators,
    calc_params_l2_norm,
    get_grad_norm
)
from teletron.core.parallel_state import get_transformer_model_group
from teletron.train.dataloader import DataloaderMixin
from teletron.models.build import build_model
from teletron.train.checkpoint import CheckPointMixin, unwrap_model, ensure_directory_exists
from teletron.train.lr_scheduler import SchedulerMixin
from teletron.train.telelogger import TeleLoggerMixin
from teletron.datasets.build import build_train_valid_test_datasets
from teletron.core.distributed.distributed_encoder import producer_process
from teletron.models.encoder_registry import get_encoder_name
from teletron.train.consumer_dataloader import create_batch_loader

from logging import getLogger
logger = getLogger(__name__)
_TRAIN_START_TIME = time.time()
ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

class Trainer(CheckPointMixin, SchedulerMixin, DataloaderMixin, TeleLoggerMixin):
    def __init__(
        self,
        args,
        dataset_provide_func=None,
    ):
        self.initialize_megatron(args)
        set_jit_fusion_options()
        transformer_group = get_transformer_model_group()
        if transformer_group is None:            
            producer_process(
                rank=dist.get_rank(), 
                world_size=dist.get_world_size(),
                encoder_name=get_encoder_name(args.model),
                device=torch.cuda.current_device(),
                build_train_valid_test_data_iterators=self.build_train_valid_test_data_iterators, 
                train_ds=None,
            )
            
            exit()        
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                        dtype=torch.double,
                                        device='cuda')
        torch.distributed.all_reduce(start_time_tensor,
                                    op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')

        self.model, self.optimizer, self.scheduler = \
                                self.setup_model_and_optimizer(args.model_type)

        self.train_itrt, self.valid_itrt, self.test_itrt = \
                                self.get_iterator(len(self.model), dataset_provide_func)

        dataiters = (self.train_itrt, self.valid_itrt, self.test_itrt)
        self.train_itrt, self.valid_itrt, self.test_itrt = [
        create_batch_loader(args, ds) if ds is not None else None 
                for ds in dataiters
            ]
        self.config = get_model_config(self.model[0])

    def setup_model_and_optimizer(self,  
                                  model_type,
                                  no_wd_decay_cond=None,
                                  scale_lr_cond=None,
                                  lr_mult=1.0):

        args = get_args()
        assert args.global_batch_size == args.micro_batch_size * mpu.get_data_parallel_world_size()
        if args.use_zero2:
            model = self.get_model(model_type, wrap_with_ddp=False)
        else:
            model = self.get_model(model_type)
        unwrapped_model = unwrap_model(model)
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = None
        if args.use_zero2:
            deepspeed.init_distributed()
            optimizer = self.get_optimizer_for_zero2(config, model, no_wd_decay_cond,
                                        scale_lr_cond, lr_mult)
        else:
            optimizer = self.get_optimizer(config, model, no_wd_decay_cond,
                                        scale_lr_cond, lr_mult)

        opt_param_scheduler = self.get_optimizer_param_scheduler(optimizer)
        if args.load is not None or args.pretrained_checkpoint is not None:
            args.iteration, args.num_floating_point_operations_so_far = self.load_checkpoint(
                model, optimizer, opt_param_scheduler, strict=True)
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0
            args.last_microbatch_size_index = None

        # get model without FP16 and/or DDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        return model, optimizer, opt_param_scheduler

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        parallel_output=True,
    ):
        args = get_args()
        cfg = core_transformer_config_from_args(args)
        return build_model(args.model, cfg)

    def get_model(self, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
        args = get_args()
        args.model_type = model_type
        if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.virtual_pipeline_model_parallel_size is not None:
            assert model_type != ModelType.encoder_and_decoder, \
                "Interleaved schedule not supported for model with both encoder and decoder"
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = self.model_provider(
                    pre_process=pre_process,
                    post_process=post_process
                )
                this_model.model_type = model_type
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = True
            add_decoder = True
            if model_type == ModelType.encoder_and_decoder:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    assert args.pipeline_model_parallel_split_rank is not None, \
                        "Split rank needs to be specified for model with both encoder and decoder"
                    rank = mpu.get_pipeline_model_parallel_rank()
                    split_rank = args.pipeline_model_parallel_split_rank
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == split_rank
                    post_process = (rank == (split_rank - 1)) or (
                            rank == (world_size - 1))
                    add_encoder = mpu.is_pipeline_stage_before_split()
                    add_decoder = mpu.is_pipeline_stage_after_split()
                model = self.model_provider(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder)
            else:
                model = self.model_provider(
                    pre_process=pre_process,
                    post_process=post_process
                )
            model.model_type = model_type

        if not isinstance(model, list):
            model = [model]

        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in model:
            for param in model_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        # GPU allocation.
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if args.fp16 or args.bf16:
            model = [Float16Module(module=model_module, config=model_module.config) for model_module in model]
        if wrap_with_ddp:
            config = get_model_config(model[0])
            model = [DDP(config,
                        model_chunk,
                        data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                        expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                        accumulate_allreduce_grads_in_fp32=args.accumulate_allreduce_grads_in_fp32,
                        overlap_grad_reduce=args.overlap_grad_reduce,
                        use_distributed_optimizer=args.use_distributed_optimizer,
                        # Turn off bucketing for model_chunk 2 onwards, since communication for these
                        # model chunks is overlapped with compute anyway.
                        disable_bucketing=(model_chunk_idx > 0),
                        check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad)
                    for (model_chunk_idx, model_chunk) in enumerate(model)]

            # Broadcast params from data parallel src rank to other data parallel ranks.
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()

        return model

    def get_iterator(
        self,
        len_model: int,
        train_valid_test_dataset_provider=None,
    ):
        args = get_args()
        if args.virtual_pipeline_model_parallel_size is not None:
            train_itrt = []
            valid_itrt = []
            test_itrt = []
            for i in range(len_model):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = self.build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider)
                train_itrt.append(iterators[0])
                valid_itrt.append(iterators[1])
                test_itrt.append(iterators[2])
        else:
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets()
            train_itrt, valid_itrt, test_itrt \
                = self.build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, 
                    train_ds_prev=train_ds,
                    valid_ds_prev=valid_ds)
        return train_itrt, valid_itrt, test_itrt



    def build_train_valid_test_data_iterators(
        self, is_tp_first=None, dp_rank=None, dp_size=None, train_ds_prev=None, valid_ds_prev=None, return_ds=False
    ):
        """Build pretraining data iterators."""

        args = get_args()

        # Build loaders.
        print("Building loaders.")
        
        if return_ds is True:
            train_dataloader, valid_dataloader, test_dataloader, train_ds, valid_ds = \
                self.build_train_valid_test_data_loaders(
                    is_tp_first,dp_rank,dp_size, train_ds_prev, valid_ds_prev, return_ds=return_ds)
        else:
            train_dataloader, valid_dataloader, test_dataloader = \
                self.build_train_valid_test_data_loaders(
                    is_tp_first,dp_rank,dp_size, train_ds_prev, valid_ds_prev)

        # Build iterators.
        print("Building iterators.")
        dl_type = args.dataloader_type

        assert dl_type in ['single', 'cyclic', 'external']

        def _get_iterator(dataloader_type, dataloader):
            """Return dataset iterator."""
            if dataloader_type == "single":
                return iter(dataloader)
            elif dataloader_type == "cyclic":
                return iter(cyclic_iter(dataloader))
            elif dataloader_type == "external":
                # External dataloader is passed through. User is expected to define how to iterate.
                return dataloader
            else:
                raise RuntimeError("unexpected dataloader type")

        if train_dataloader is not None:
            train_data_iterator = _get_iterator(dl_type, train_dataloader)
        else:
            train_data_iterator = None

        if valid_dataloader is not None:
            valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
        else:
            valid_data_iterator = None

        if test_dataloader is not None:
            test_data_iterator = _get_iterator(dl_type, test_dataloader)
        else:
            test_data_iterator = None

        if return_ds is True:
            return train_data_iterator, valid_data_iterator, test_data_iterator, train_ds, valid_ds
        else:
            return train_data_iterator, valid_data_iterator, test_data_iterator

    def initialize_megatron(self, args):

        if args.distributed_vae:
            args.world_size = (args.world_size - args.distributed_vae_world_size)  //args.consumer_models_num
            args.dit_world_size = args.world_size * args.consumer_models_num
        validate_args(args)
        set_args(args)

        if args.distributed_vae:
            args.world_size = args.distributed_vae_world_size + args.dit_world_size
        def finish_mpu_init():
            args = get_args()
            _initialize_distributed()
            if args.rank == 0:
                print("> setting random seeds to {} ...".format(args.seed))

            from teletron.core.parallel_state import get_transformer_model_group
            isDiTRank = get_transformer_model_group()
            if isDiTRank is not None:
                _set_random_seed(args.seed, args.data_parallel_random_init)
        args = get_args()

        if args.lazy_mpu_init:
            args.use_cpu_initialization = True
            # delayed initialization of DDP-related stuff
            # We only set basic DDP globals
            mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            # and return function for external DDP manager
            # to call when it has DDP initialized
            mpu.set_tensor_model_parallel_rank(args.rank)
            return finish_mpu_init
        else:
            # Megatron's MPU is the master. Complete initialization right away.
            finish_mpu_init()
            # Autoresume.
            # _init_autoresume()
            # Compile dependencies.
            from teletron.core.parallel_state import get_transformer_model_group
            isConsumerRank = get_transformer_model_group()
            if isConsumerRank is not None:
                _compile_dependencies()
            if args.tp_comm_overlap:
                _initialize_tp_communicators()
            # No continuation function
            return None

    def pretrain(
        self,
        forward_step_func=forward_step,
        process_non_loss_data_func=None,
    ):
        args = get_args()

        if args.distributed_vae:
            consumer_config = torch.zeros(
                (3), dtype=torch.int64, device=torch.cuda.current_device()
            )
            consumer_config[0] = args.iteration
            consumer_config[1] = args.consumed_train_samples
            consumer_config[2] = args.consumed_valid_samples

            from teletron.core.parallel_state import get_comm_pair
            comm_pair = get_comm_pair()

            if comm_pair is not None:
                req = dist.isend(tensor=consumer_config, dst=comm_pair.producer, tag=0)
                req.wait()
        print_datetime('after dataloaders are built')
        print_rank_0('done with setup ...')

        if not args.skip_train:
            print_rank_0('training ...')

            if args.dataloader_type == 'cyclic' and args.retro_project_dir:
                assert args.retro_cyclic_train_iters is not None
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = self.train(
                    forward_step_func,
                    # forward_step_func,
                    self.model, self.optimizer, self.scheduler,
                    self.train_itrt, self.valid_itrt,
                    process_non_loss_data_func, self.config)

            print_datetime('after training is done')

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                self.save_checkpoint(iteration, self.model, self.optimizer, self.scheduler,
                                num_floating_point_operations_so_far)
        else:
            print_rank_0('skipping training (--skip-train is on) ...')
            iteration = args.iteration

        if args.do_valid:
            prefix = f'iteration {iteration} on validation set'
            self.evaluate_and_print_results(prefix, forward_step_func,
                                    self.valid_itrt, self.model,
                                    iteration, process_non_loss_data_func, self.config,
                                    verbose=True, write_to_tensorboard=not args.skip_train)

        if args.do_test:
            prefix = f'iteration {iteration} on test set'
            self.evaluate_and_print_results(prefix, forward_step_func,
                                    self.test_itrt, self.model,
                                    iteration, process_non_loss_data_func, self.config,
                                    verbose=True, write_to_tensorboard=not args.skip_train)

    def train(
        self,
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
    ):
        args = get_args()

        for model_module in model:
            model_module.train()
        total_loss_dict = {}

        # Iterations.
        iteration = args.iteration

        num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

        # Setup some training config params
        # config.grad_scale_func = self.optimizer.scale_loss
        if isinstance(model[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, \
                ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
            if len(model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
                if len(model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.delay_param_gather:
            config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                    for model_index in range(len(model))]
            if len(model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads

        print_datetime('before the start of training step')
        report_memory_flag = True
        exit = False

        if args.manual_gc:
            # Disable the default garbage collector and perform the collection manually.
            # This is to align the timing of garbage collection across ranks.
            assert args.manual_gc_interval >= 0, \
                'Manual garbage collection interval should be laerger than or equal to 0.'
            gc.disable()
            gc.collect()

        num_microbatches = get_num_microbatches()
        eval_duration = 0.0
        eval_iterations = 0
        
        if args.consumer_profile:
            prof_save_path = os.path.join(args.profile_path, f"consumer/rank_{dist.get_rank()}.json")
            ensure_directory_exists(prof_save_path)
            def trace_handler(p):
                p.export_chrome_trace(prof_save_path)

            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
                on_trace_ready=trace_handler,
                record_shapes=True
            )

        while iteration < args.train_iters:
            if args.consumer_profile and iteration == args.profile_step_start:
                prof.start()
            if args.consumer_profile and iteration == args.profile_step_end:
                prof.stop()
            if args.profile and \
            iteration == args.profile_step_start and \
            torch.distributed.get_rank() in args.profile_ranks:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

            # Update number of microbatches first without consistency check to decide if a
            # checkpoint should be saved. If the number of microbatches is different
            # from the previous iteration, save a checkpoint. Then run consistency check
            # to make sure training configuration is still valid.
            update_num_microbatches(args.consumed_train_samples, consistency_check=False)
            if get_num_microbatches() != num_microbatches and iteration != 0:
                assert get_num_microbatches() > num_microbatches, \
                    "number of microbatches should be increasing due to batch size rampup"
                self.save_checkpoint_and_time(iteration, model, optimizer,
                                        opt_param_scheduler,
                                        num_floating_point_operations_so_far)
            num_microbatches = get_num_microbatches()
            update_num_microbatches(args.consumed_train_samples, consistency_check=True)

            args.curr_iteration = iteration
            
            if os.environ.get("MEMORY_SNAPSHOT"):
                torch.cuda.memory._record_memory_history(max_entries=80000)
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                self.train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config)
            
            if grad_norm is None:
                if args.use_zero2:
                    grad_norm = optimizer._global_grad_norm
                else:
                    grad_norm = get_grad_norm(optimizer)
                
            if os.environ.get("MEMORY_SNAPSHOT"):
                time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                save_dir = os.environ.get("PROF_SAVE_PATH", ".")  # 默认当前目录
                file_name = os.path.join(save_dir, f"memory_{time_str}_iter{iteration}_rank{torch.distributed.get_rank()}.pt")
                torch.cuda.memory._dump_snapshot(file_name)
                torch.cuda.memory._record_memory_history(enabled=None)
            iteration += 1
            batch_size = mpu.get_data_parallel_world_size() * \
                        args.micro_batch_size * \
                        get_num_microbatches()
            args.consumed_train_samples += batch_size
            num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)

            # Logging.
            if args.use_zero2:
                loss_scale = optimizer._get_loss_scale()
            else:
                loss_scale = optimizer.get_loss_scale().item()
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)

            learning_rate = None
            decoupled_learning_rate = None
            for param_group in optimizer.param_groups:
                if param_group['is_decoupled_lr']:
                    decoupled_learning_rate = param_group['lr']
                else:
                    learning_rate = param_group['lr']

            report_memory_flag = self.log_training_infos(
                loss_dict, total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                iteration, loss_scale,
                report_memory_flag, skipped_iter,
                grad_norm, params_norm, num_zeros_in_grad
            )

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and \
                    args.do_valid:
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    optimizer.disable_pre_hook()
                if args.manual_gc and args.manual_gc_eval:
                    # Collect all objects.
                    gc.collect()
                prefix = 'iteration {}'.format(iteration)
                self.evaluate_and_print_results(prefix, forward_step_func,
                                        valid_data_iterator, model,
                                        iteration, process_non_loss_data_func,
                                        config, False)
                eval_iterations += args.eval_iters
                if args.manual_gc and args.manual_gc_eval:
                    # Collect only the objects created and used in evaluation.
                    gc.collect(generation=0)
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    optimizer.enable_pre_hook()

            # Checkpointing
            saved_checkpoint = False
            if args.save and args.save_interval and \
                            iteration % args.save_interval == 0:
                self.save_checkpoint_and_time(iteration, model, optimizer,
                                        opt_param_scheduler,
                                        num_floating_point_operations_so_far)
                saved_checkpoint = True

            # Exiting based on duration
            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    if not saved_checkpoint:
                        self.save_checkpoint_and_time(iteration, model, optimizer,
                                                opt_param_scheduler,
                                                num_floating_point_operations_so_far)
                    print_datetime('exiting program after {} minutes'.format(train_time))
                    exit = True
                    break

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if args.save and not saved_checkpoint:
                    self.save_checkpoint_and_time(iteration, model, optimizer,
                                            opt_param_scheduler,
                                            num_floating_point_operations_so_far)
                torch.distributed.barrier()
                print_datetime('exiting program at iteration {}'.format(iteration))
                exit = True
                break

            if args.profile and \
            iteration == args.profile_step_end and \
            torch.distributed.get_rank() in args.profile_ranks:
                torch.cuda.cudart().cudaProfilerStop()

            if args.manual_gc:
                if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                    gc.collect()

        # Close out pre-hooks if using distributed optimizer and overlapped param gather.
        if args.use_distributed_optimizer and args.overlap_param_gather:
            optimizer.disable_pre_hook()

        # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
        if exit:
            sys.exit()

        return iteration, num_floating_point_operations_so_far

    def train_step(
        self,
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
    ):
        """Single training step."""
        args = get_args()

        if not args.use_zero2:
            for model_chunk in model:
                model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        if args.use_zero2:
            losses_reduced = deepspeed_forward_backward(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                forward_only=False,
                zero_optimizer=optimizer)
        else:
            forward_backward_func = get_forward_backward_func()
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=False)

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.
        if args.use_zero2:
            optimizer.step()
            update_successful = True
            grad_norm = None
            num_zeros_in_grad = None
        else:
            update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Vision momentum.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def evaluate_and_print_results(
        self,
        prefix,
        forward_step_func,
        data_iterator,
        model,
        iteration,
        process_non_loss_data_func,
        config,
        verbose=False,
        write_to_tensorboard=True,
    ):
        """Helper function to evaluate and dump results on screen."""
        args = get_args()

        total_loss_dict, collected_non_loss_data, timelimit = self.evaluate(
            forward_step_func, data_iterator, model,
            process_non_loss_data_func, config, verbose)
        # Timelimit hit during evaluation
        if timelimit:
            return
        string = ' validation loss at {} | '.format(prefix)
        import math
        for key in total_loss_dict:
            string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
            ppl = math.exp(min(20, total_loss_dict[key].item()))
            string += '{} PPL: {:.6E} | '.format(key, ppl)

        length = len(string) + 1
        print_rank_last('-' * length)
        print_rank_last(string)
        print_rank_last('-' * length)

        self.log_validation_infos(total_loss_dict, iteration)

    def evaluate(
        self,
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose=False,
    ):
        """Evaluation."""
        args = get_args()

        # Turn on evaluation mode which disables dropout.
        for model_module in model:
            model_module.eval()

        total_loss_dict = {}

        # make validation batch size independent from training batch size
        eval_batch_size = args.global_batch_size
        eval_num_microbatches = eval_batch_size // \
            (args.micro_batch_size * args.data_parallel_size)

        with torch.no_grad():
            iteration = 0
            if verbose:
                print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
            while iteration < args.eval_iters:
                iteration += 1
                if verbose:
                    print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

                forward_backward_func = get_forward_backward_func()
                # Don't care about timing during evaluation
                config.timers = None
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)

                # Empty unused memory
                if args.empty_unused_memory_level >= 1:
                    torch.cuda.empty_cache()

                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # Reduce across processes.
                    for loss_dict in loss_dicts:
                        for key in loss_dict:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]

                args.consumed_valid_samples += eval_batch_size

                if args.exit_duration_in_mins:
                    train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                    done_cuda = torch.tensor(
                        [train_time > args.exit_duration_in_mins],
                        dtype=torch.int, device='cuda')
                    torch.distributed.all_reduce(
                        done_cuda, op=torch.distributed.ReduceOp.MAX)
                    done = done_cuda.item()
                    if done:
                        print_rank_0('Exiting during evaluation, timelimit reached')
                        return None, None, True

            collected_non_loss_data = None
            if process_non_loss_data_func is not None and is_last_rank():
                collected_non_loss_data = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                    collect_non_loss_data=True)

        # Move model back to the train mode.
        for model_module in model:
            model_module.train()

        for key in total_loss_dict:
            total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

        return total_loss_dict, collected_non_loss_data, False
