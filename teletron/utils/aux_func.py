import torch
from types import SimpleNamespace
import os
from datetime import datetime
import pathlib
import subprocess
from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""




def validate_yaml(args, defaults={}):
    
    # This is for legacy script env var setting
    if type(args.data_path) is str:
        # If no white space its a single path
        split_data_path = args.data_path.split()
        if len(split_data_path) != 1:
            args.data_path = split_data_path

    # Tensor model parallel size.
    args.model_parallel.tensor_model_parallel_size = min(
        args.model_parallel.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.model_parallel.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.model_parallel.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.model_parallel.pipeline_model_parallel_size = min(
        args.model_parallel.pipeline_model_parallel_size,
        (args.world_size // args.model_parallel.tensor_model_parallel_size))
    args.model_parallel.transformer_pipeline_model_parallel_size = (
        args.model_parallel.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.model_parallel.pipeline_model_parallel_size
    )
    # Checks.
    model_parallel_size = args.model_parallel.pipeline_model_parallel_size * \
                          args.model_parallel.tensor_model_parallel_size
    assert args.world_size % (model_parallel_size * args.model_parallel.context_parallel_size) == 0, \
        'world size ({}) is not divisible by tensor parallel size ({}) times ' \
        'pipeline parallel size ({}) times context parallel size ({})'.format(
        args.world_size, args.model_parallel.tensor_model_parallel_size,
        args.model_parallel.pipeline_model_parallel_size, args.model_parallel.context_parallel_size)
    
    # data_parallel_size is not in model parallel config
    args.data_parallel_size = args.world_size // (model_parallel_size * args.model_parallel.context_parallel_size)
    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {} '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.model_parallel.context_parallel_size,
                  args.model_parallel.tensor_model_parallel_size,
                  args.model_parallel.pipeline_model_parallel_size), flush=True)
    if args.model_parallel.pipeline_model_parallel_size > 1:
        if args.model_parallel.pipeline_model_parallel_split_rank is not None:
            assert args.model_parallel.pipeline_model_parallel_split_rank < \
                    args.model_parallel.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.model_parallel.pipeline_model_parallel_size)

    if args.model_parallel.tp_comm_overlap:
        assert args.model_parallel.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0

    # num_layers_per_virtual_pipeline_stage is not insde model parallel for checkpointing
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.model_parallel.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.language_model.num_layers % args.model_parallel.transformer_pipeline_model_parallel_size == 0, \
            'number of layers should be divisible by the pipeline parallel size'
        num_layers_per_pipeline_stage = args.language_model.num_layers // args.model_parallel.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
        args.model_parallel.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.model_parallel.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.model_parallel.overlap_p2p_comm = False
        if args.rank == 0:
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '
                  'schedule does not support overlapping p2p communication')

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            '--overlap-grad-reduce should be turned on when using --overlap-param-gather'

    # Parameters dtype.
    if args.model_parallel.fp16:
        assert not args.model_parallel.bf16
        args.model_parallel.params_dtype = torch.half
    if args.model_parallel.bf16:
        assert not args.model_parallel.fp16
        args.model_parallel.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.model_parallel.params_dtype),
              flush=True)


    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.model_parallel.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    # How to handle this better
    if args.language_model.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.language_model.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.language_model.num_layers = args.encoder_num_layers

    # Check required arguments.
    # removed max_position_embeddings from reqs
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads']
    for req_arg in required_args:
        _check_arg_is_not_none(args.language_model, req_arg)

    # Checks.
    if args.language_model.ffn_hidden_size is None:
        if args.language_model.activation_func == "swiglu":
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.language_model.ffn_hidden_size = int((4 * args.language_model.hidden_size * 2 / 3) / 64) * 64
        else:
            args.language_model.ffn_hidden_size = 4 * args.language_model.hidden_size

    if args.language_model.kv_channels is None:
        assert args.language_model.hidden_size % args.language_model.num_attention_heads == 0
        args.language_model.kv_channels = args.language_model.hidden_size // args.language_model.num_attention_heads

    #TODO: Implement arguments for encoder-decoder
    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.language_model.fp32_residual_connection:
        assert args.model_parallel.fp16 or args.model_parallel.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.language_model.moe_grouped_gemm:
        assert args.model_parallel.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
        dc = torch.cuda.get_device_capability()
        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.language_model.persist_layer_norm = False
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.language_model.distribute_saved_activations:
        assert args.model_parallel.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.language_model.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.language_model.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert (TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    if args.language_model.recompute_granularity == 'selective':
        assert args.language_model.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.model_parallel.tensor_model_parallel_size == 1:
        args.model_parallel.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.model_parallel.sequence_parallel:
        args.model_parallel.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.model_parallel.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.model_parallel.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Retro checks.
    if getattr(args, 'retro_add_retriever', False):
        raise Exception("Retro untested for yaml args. See arguments.py.")

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    #TODO: Retro args loading not tested
    # Load retro args (used by both Retro & GPT).
    if getattr(args, 'retro_project_dir', None) is not None:
        raise Exception("Retro untested for yaml args. See arguments.py.")

    if args.language_model.rotary_interleaved and args.language_model.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    
    # MoE Spec check
    if args.language_model.num_moe_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"
        if args.model_parallel.tensor_model_parallel_size > 1:
            assert args.model_parallel.sequence_parallel, \
                "When using MoE and tensor parallelism, sequence parallelism must be used."

    # Expert parallelism check
    if args.model_parallel.expert_model_parallel_size  > 1:
        assert args.language_model.num_moe_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.language_model.num_moe_experts % args.model_parallel.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.model_parallel.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # Print arguments.
    _print_args("arguments", args)

    #TODO: Added as much of the global initialization requires the model parallel arguments
    args = SimpleNamespace(**args.__dict__, **args.model_parallel.__dict__)
    args = SimpleNamespace(**args.__dict__, **args.language_model.__dict__)
    # For GPT Layer spec in pretrain_gpt
    args.num_experts = args.language_model.num_moe_experts

    return args


def validate_args(args, defaults={}):

    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)

    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )

    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % (model_parallel_size * args.context_parallel_size) == 0, \
        'world size ({}) is not divisible by tensor parallel size ({}) times ' \
        'pipeline parallel size ({}) times context parallel size ({})'.format(
        args.world_size, args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size, args.context_parallel_size)
    args.data_parallel_size = args.world_size // (model_parallel_size * args.context_parallel_size)
    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {} '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    if args.tp_comm_overlap:
        assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
            'number of layers should be divisible by the pipeline parallel size'
        num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        if args.rank == 0:
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '
                  'schedule does not support overlapping p2p communication')

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            '--overlap-grad-reduce should be turned on when using --overlap-param-gather'
        assert args.use_mcore_models, \
            '--overlap-param-gather only supported with MCore models'

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)
        
    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert (TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        assert args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config."

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert args.use_mcore_models, \
            '--decoupled-lr and --decoupled-min-lr only supported by Megatron Core, please add --use-mcore-models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and not args.use_mcore_models:
        raise RuntimeError('--rotary-interleaved only support Megatron Core, please add --use-mcore-models.')

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # MoE Spec check
    if args.num_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"
        if args.tensor_model_parallel_size > 1:
            assert args.sequence_parallel, \
                "When using MoE and tensor parallelism, sequence parallelism must be used."

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and not args.use_mcore_models:
        raise RuntimeError('--use-dist-ckpt only support Megatron Core, please add --use-mcore-models.')

    if args.use_zero2:
        assert args.use_dist_ckpt == False, ("When using DeepspeedZero2, dist ckpt format is not supported, please set --use-dist-ckpt False")
    # Print arguments.
    _print_args("arguments", args)

    return args


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)

def _print_args(title, args):
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)
        
def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    """Get an attribute from a wrapped model.
    If return_model_obj is true, return the object that has the 'attr' attribute;
    otherwise, return the attribute directly."""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    # breakpoint()
    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else: # here

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if hasattr(model, "module"):
            model = model.module
        # elif hasattr(model, "modules"):
        #     model = model.modules
        else:
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        

    if return_model_obj:
        return model
    return getattr(model, attr)

def num_floating_point_operations(args, batch_size):
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
        12
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            + (args.num_query_groups / args.num_attention_heads)
            + (args.seq_length / args.hidden_size)
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )


def get_model_config(model):
    # # Prefer the megatron/transformer config if the model exposes it
    # cfg = get_attr_wrapped_model(model, 'parallel_config', allow_none=True)
    # if cfg is not None:
    #     return cfg
    # # Fallback to default 'config' (may be HF FrozenDict on some models)
    # return get_attr_wrapped_model(model, 'config', allow_none=False)
    return get_attr_wrapped_model(model, 'config')


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def fused_kernel_load(args):

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
        cpp_extension.CUDA_HOME
    )
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')
        if int(bare_metal_minor) >= 8:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=(args.rank == 0),
        )


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
