import math
from typing import Callable, List, Optional
from teletron.utils import print_rank_0, get_args, get_num_microbatches
from teletron.train.utils import update_train_iters
from megatron.core import mpu
from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    _update_min_and_max_lr_in_param_groups,
    _get_megatron_optimizer_based_on_param_groups,
    ChainedOptimizer,
)
from megatron.core.transformer.module import MegatronModule
from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils.timer import NoopTimer

import torch
from logging import getLogger

logger = getLogger(__name__)


class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay"""

    def __init__(self, optimizer, init_lr, max_lr, min_lr,
                 lr_warmup_steps, lr_decay_steps, lr_decay_style,
                 start_wd, end_wd, wd_incr_steps, wd_incr_style,
                 use_checkpoint_opt_param_scheduler=True,
                 override_opt_param_scheduler=False):

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, 'both override and '\
                'use-checkpoint are set.'

        # Set the learning rate
        self.step(0)
        print_rank_0('> learning rate decay style: {}'.format(self.lr_decay_style))


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == 'constant':
            assert self.start_wd == self.end_wd
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception('{} weight decay increment style is not supported.'.format(
                self.wd_incr_style))

        return self.start_wd + coeff * delta_wd


    def get_lr(self, param_group):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        max_lr = self.max_lr 
        min_lr = self.min_lr
        
        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return (
                self.init_lr
                + (
                    (max_lr - self.init_lr)
                    * float(self.num_steps)
                    / float(self.lr_warmup_steps)
                )
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps ** 0.5 / (num_steps ** 0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.lr_decay_style))

        return min_lr + coeff * delta_lr


    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group['lr'] = new_lr * param_group.get('lr_mult', 1.0)
            param_group['weight_decay'] = new_wd * param_group.get('wd_mult', 1.0)


    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, \
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')

        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')

        if 'warmup_iter' in sd:
            lr_warmup_steps_ = sd['warmup_iter']
        elif 'warmup_steps' in sd:
            lr_warmup_steps_ = sd['warmup_steps']
        else:
            lr_warmup_steps_ = sd['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps,
                                                lr_warmup_steps_,
                                                'warmup iterations')

        if 'end_iter' in sd:
            lr_decay_steps_ = sd['end_iter']
        elif 'decay_steps' in sd:
            lr_decay_steps_  = sd['decay_steps']
        else:
            lr_decay_steps_ = sd['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, lr_decay_steps_,
                                               'total number of iterations')

        if 'decay_style' in sd:
            lr_decay_style_ = sd['decay_style']
        else:
            lr_decay_style_ = sd['lr_decay_style']
        self.lr_decay_style = self._check_and_set(self.lr_decay_style,
                                               lr_decay_style_,
                                               'learning rate decay style')

        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        self.step(increment=num_steps)


        if 'start_wd' in sd:
            self.start_wd = self._check_and_set(self.start_wd,
                                                sd['start_wd'],
                                                "start weight decay")
            self.end_wd = self._check_and_set(self.end_wd,
                                                sd['end_wd'],
                                                "end weight decay")
            self.wd_incr_steps = self._check_and_set(self.wd_incr_steps,
                                                sd['wd_incr_steps'],
                                                "total number of weight decay iterations")
            self.wd_incr_style = self._check_and_set(self.wd_incr_style,
                                                sd['wd_incr_style'],
                                                "weight decay incr style")

class SchedulerMixin:

    def get_optimizer_param_scheduler(self, optimizer):
        """Build the learning rate scheduler."""
        args = get_args()

        # Iteration-based training.
        if args.train_iters:
            if args.lr_decay_iters is None:
                args.lr_decay_iters = args.train_iters
            lr_decay_steps = args.lr_decay_iters * args.global_batch_size
            wd_incr_steps = args.train_iters * args.global_batch_size
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
        # Sample-based training.
        elif args.train_samples:
            # We need to set training iters for later use. Technically
            # we need to adjust the training samples too (due to last
            # batch being incomplete) but we leave it as is for now.
            update_train_iters(args)
            if args.lr_decay_samples is None:
                args.lr_decay_samples = args.train_samples
            lr_decay_steps = args.lr_decay_samples
            wd_incr_steps = args.train_samples
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_samples
        else:
            raise Exception(
                'either train-iters or train-samples should be provided.')

        opt_param_scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=args.lr_warmup_init,
            max_lr=args.lr,
            min_lr=args.min_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            start_wd=args.start_weight_decay,
            end_wd=args.end_weight_decay,
            wd_incr_steps=wd_incr_steps,
            wd_incr_style=args.weight_decay_incr_style,
            use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=args.override_opt_param_scheduler)

        return opt_param_scheduler

    def get_scheduler(self, optimizer):
        args = get_args()
        # Iteration-based training.
        if args.train_iters:
            if args.lr_decay_iters is None:
                args.lr_decay_iters = args.train_iters
            lr_decay_steps = args.lr_decay_iters * args.global_batch_size
            wd_incr_steps = args.train_iters * args.global_batch_size
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
        # Sample-based training.
        elif args.train_samples:
            # We need to set training iters for later use. Technically
            # we need to adjust the training samples too (due to last
            # batch being incomplete) but we leave it as is for now.
            update_train_iters(args)
            if args.lr_decay_samples is None:
                args.lr_decay_samples = args.train_samples
            lr_decay_steps = args.lr_decay_samples
            wd_incr_steps = args.train_samples
            if args.lr_warmup_fraction is not None:
                lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
            else:
                lr_warmup_steps = args.lr_warmup_samples
        else:
            raise Exception(
                'either train-iters or train-samples should be provided.')

        opt_param_scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=args.lr_warmup_init,
            max_lr=args.lr,
            min_lr=args.min_lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=args.lr_decay_style,
            start_wd=args.start_weight_decay,
            end_wd=args.end_weight_decay,
            wd_incr_steps=wd_incr_steps,
            wd_incr_style=args.weight_decay_incr_style,
            use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=args.override_opt_param_scheduler)

        return opt_param_scheduler

    def get_optimizer_for_zero2(self, config: OptimizerConfig, 
                        model: List[MegatronModule], 
                        no_weight_decay_cond: Optional[Callable] = None, 
                        scale_lr_cond: Optional[Callable] = None, 
                        lr_mult: float = 1.0,):
        args = get_args()
        # Collect param groups.
        param_groups = _get_param_groups(
            model,
            no_weight_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_decoupled_learning_rate=config.decoupled_lr is not None,
        )
        param_groups = _update_min_and_max_lr_in_param_groups(
            param_groups,
            lr=config.lr,
            min_lr=config.min_lr,
            decoupled_lr=config.decoupled_lr,
            decoupled_min_lr=config.decoupled_min_lr,
        )
        if config.optimizer == 'adam':
            base_optimizer = Adam(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
            )
        elif config.optimizer == 'sgd':
            base_optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
        param_names = {param:name for name, param in model[0].named_parameters()}
        timers = NoopTimer()
        optimizer = DeepSpeedZeroOptimizer(
                base_optimizer,
                param_names,
                timers=timers,
                static_loss_scale=1.0,
                dynamic_loss_scale=False,
                dynamic_loss_args=None,
                clip_grad=args.clip_grad,
                contiguous_gradients=True,
                reduce_bucket_size=500000000,
                use_multi_rank_bucket_allreduce=True,
                allgather_bucket_size=500000000,
                dp_process_group=mpu.get_data_parallel_group(with_context_parallel=True),
                expert_parallel_group=None,
                expert_data_parallel_group=None,
                reduce_scatter=True,
                overlap_comm=False,
                offload_optimizer_config=None,
                mpu=None,
                postscale_gradients=True,
                gradient_predivide_factor=1.0,
                gradient_accumulation_steps=get_num_microbatches(),
                ignore_unused_parameters=True,
                partition_grads=True,
                round_robin_gradients=False,
                has_moe_layers=False,
                fp16_master_weights_and_gradients=False,
                elastic_checkpoint=False,)
        return optimizer

    def get_optimizer(self, config: OptimizerConfig, 
                        model: List[MegatronModule], 
                        no_weight_decay_cond: Optional[Callable] = None, 
                        scale_lr_cond: Optional[Callable] = None, 
                        lr_mult: float = 1.0):
        
        """Retrieve the Megatron optimizer for model chunks.

        We use separate optimizers for expert parameters and non-expert parameters.

        Args:
            config (OptimizerConfig): optimizer configuration object.
            model_chunks (List[MegatronModule]): model chunks to get optimizer for.
            no_weight_decay_cond (func, optional): function to determine whether a parameter
                should not perform weight decay. Defaults to None.
            scale_lr_cond (func, optional): function to determine whether a parameter
                should have a scaled learning rate. Defaults to None.
            lr_mult (float, optional): learning rate multiplier for parameters that
                satisfy scale_lr_cond. Defaults to 1.0.

        Returns:
            Instance of MegatronOptimizer.
        """

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f'Setting up optimizer with {config}')

        # Collect param groups.
        param_groups = _get_param_groups(
            model,
            no_weight_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_decoupled_learning_rate=config.decoupled_lr is not None,
        )
        param_groups = _update_min_and_max_lr_in_param_groups(
            param_groups,
            lr=config.lr,
            min_lr=config.min_lr,
            decoupled_lr=config.decoupled_lr,
            decoupled_min_lr=config.decoupled_min_lr,
        )

        # Collect grad buffers for distributed optimizer.
        per_model_buffers = {}
        per_model_ep_buffers = {}
        for model_idx, model_chunk in enumerate(model):
            if hasattr(model_chunk, 'buffers'):
                per_model_buffers[model_idx] = model_chunk.buffers
                per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers

        # Split param groups into dense and MoE params (since data-parallel groups for MoE
        # parameters can be different with expert parallelism).
        dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
        moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))

        # Create optimizers.
        model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())
        optimizers = [
            _get_megatron_optimizer_based_on_param_groups(
                config,
                param_groups=dense_param_groups,
                per_model_buffers=per_model_buffers,
                data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
                data_parallel_group_idx=model_parallel_rank,
            )
        ]
        if len(moe_param_groups) > 0:
            model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
            expert_parallel_rank = mpu.get_expert_model_parallel_rank()
            optimizers.append(
                _get_megatron_optimizer_based_on_param_groups(
                    config,
                    param_groups=moe_param_groups,
                    per_model_buffers=per_model_ep_buffers,
                    data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                    data_parallel_group_gloo=mpu.get_data_modulo_expert_parallel_group_gloo(),
                    data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size
                    + model_parallel_rank,
                )
            )

        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)
