# import os
# import socket
# import time
# import glob
# import torch
# import gc
# import queue as pyqueue
# import multiprocessing as mp
# import tempfile
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from unittest import TestCase
# from unittest.mock import patch
# from unit_tests.test_utils import spawn
# from megatron.core import mpu
# from teletron.train.checkpoint.checkpoint import CheckPointMixin


# def _load_zero2_file_gpu(args):
#     import torch as _torch
#     path, device = args
#     if not _torch.cuda.is_available():
#         raise RuntimeError("CUDA unavailable in zero2 shard loader subprocess")
#     _torch.cuda.set_device(device)
#     return _torch.load(path, map_location=f"cuda:{device}", weights_only=False)


# def _load_zero2_file_cpu(path: str):
#     import torch as _torch
#     return _torch.load(path, map_location="cpu", weights_only=False)


# def _move_to_cuda(obj, device: int):
#     if torch.is_tensor(obj):
#         return obj.to(device=f"cuda:{device}", non_blocking=True)
#     if isinstance(obj, dict):
#         return {k: _move_to_cuda(v, device) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [_move_to_cuda(v, device) for v in obj]
#     if isinstance(obj, tuple):
#         return tuple(_move_to_cuda(v, device) for v in obj)
#     return obj


# def _find_first_tensor_device(obj):
#     if torch.is_tensor(obj):
#         return str(obj.device)
#     if isinstance(obj, dict):
#         for v in obj.values():
#             d = _find_first_tensor_device(v)
#             if d is not None:
#                 return d
#     if isinstance(obj, list) or isinstance(obj, tuple):
#         for v in obj:
#             d = _find_first_tensor_device(v)
#             if d is not None:
#                 return d
#     return None


# class _RealZero2Optimizer:
#     def __init__(self, rank: int, q):
#         self.rank = rank
#         self.q = q
#         self.called = False
#         self.loaded_shard_indices = []
#         self.loaded_shard_keys = {}

#     def load_state_dict(self, state_dict, load_from_fp32_weights=False):
#         self.called = True
#         if not isinstance(state_dict, list):
#             raise TypeError(f"expected list for zero2 optimizer state, got {type(state_dict)}")

#         loaded = []
#         keys_by_idx = {}
#         for i, shard in enumerate(state_dict):
#             if shard is None:
#                 continue
#             if not isinstance(shard, dict):
#                 raise TypeError(f"expected shard dict at index {i}, got {type(shard)}")
#             loaded.append(i)
#             try:
#                 keys_by_idx[i] = sorted(list(shard.keys()))[:20]
#             except Exception:
#                 keys_by_idx[i] = []
#         self.loaded_shard_indices = loaded
#         self.loaded_shard_keys = keys_by_idx
#         first_device = _find_first_tensor_device(state_dict)
#         if first_device is not None:
#             self.q.put(f"optimizer_real_first_tensor_device rank{self.rank} device={first_device}")
#         self.q.put(
#             f"optimizer_real_loaded rank{self.rank} loaded_shards={len(loaded)} "
#             f"load_from_fp32_weights={bool(load_from_fp32_weights)}"
#         )


# def _parallel_deepspeed_zero2_load_checkpoint(rank, world_size, q):
#     import datetime
#     import types
#     import random
#     import numpy as np
#     import torch.nn as nn
#     from unittest.mock import patch as _patch

#     if "CUDA_VISIBLE_DEVICES" not in os.environ:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#     if not torch.cuda.is_available():
#         q.put(f"cuda_unavailable rank{rank}")
#         return
#     device_count = torch.cuda.device_count()
#     if device_count == 0:
#         q.put(f"no_cuda_devices_visible rank{rank}")
#         return

#     cuda_rank = rank % device_count
#     torch.cuda.set_device(cuda_rank)

#     try:
#         import deepspeed
#         from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
#         from deepspeed.utils.timer import NoopTimer
#     except Exception as e:
#         q.put(f"deepspeed_unavailable rank{rank} error={str(e)}")
#         return

#     load_dir = os.environ.get("DS_ZERO2_LOAD_CKPT_DIR", "")
#     if not load_dir:
#         q.put(f"deepspeed_no_dir rank{rank}")
#         return
#     iteration = int(os.environ.get("DS_ZERO2_LOAD_CKPT_ITER", "1"))
    
#     # Ensure 16 threads per rank for loading
#     os.environ["ZERO2_LOAD_THREADS"] = "16"
#     os.environ["ZERO2_SUBPROC_PER_RANK"] = "16"

#     try:
#         torch.distributed.init_process_group(
#             backend="nccl",
#             init_method="env://",
#             world_size=world_size,
#             rank=rank,
#             timeout=datetime.timedelta(minutes=3),
#         )
#         deepspeed.init_distributed()

#         from teletron.core.parallel_state import initialize_model_parallel_base
#         from teletron.train.checkpoint.utils import (
#             get_checkpoint_name,
#             get_checkpoint_tracker_filename,
#         )

#         from unittest.mock import patch as __patch, Mock as __Mock
#         from dataclasses import dataclass, asdict
#         from typing import Tuple
#         from megatron.core.transformer import TransformerConfig as __TransformerConfig
#         from megatron.core import mpu as __mpu

#         @dataclass
#         class _TeleaiParams:
#             dim: int = 5120
#             in_dim: int = 36
#             out_dim: int = 16
#             text_dim: int = 4096
#             freq_dim: int = 256
#             ffn_dim: int = 13824
#             eps: float = 1e-6
#             patch_size: Tuple[int, int, int] = (1, 2, 2)
#             num_heads: int = 40
#             num_layers: int = 1
#             has_image_input: bool = True
#             has_image_pos_emb: bool = False

#         with __patch("teletron.utils.set_config") as __mock_set_config, __patch(
#             "teletron.utils.get_args"
#         ) as __mock_get_args:
#             from teletron.models.teleai import ParallelTeleaiModel, TeleaiModel

#             __args = __Mock()
#             __args.recompute_method = "block"
#             __args.recompute_granularity = "full"
#             __args.recompute_num_layers = 1
#             __args.activation_offload = True
#             __args.num_layers = 1
#             __args.num_attention_heads = 40
#             __args.distributed_vae = False
#             __args.consumer_models_num = 1
#             __mock_get_args.return_value = __args

#             __model_config = dict(
#                 dit=dict(
#                     type="ParallelTeleaiModel",
#                     config=dict(
#                         has_image_input=True,
#                         patch_size=[1, 2, 2],
#                         in_dim=36,
#                         dim=5120,
#                         ffn_dim=13824,
#                         freq_dim=256,
#                         text_dim=4096,
#                         out_dim=16,
#                         num_heads=40,
#                         num_layers=1,
#                         eps=1e-6,
#                         has_image_pos_emb=False,
#                     ),
#                 )
#             )
#             __mock_set_config.return_value = {"model_config": __model_config}

#             initialize_model_parallel_base(
#                 tensor_model_parallel_size=1,
#                 pipeline_model_parallel_size=1,
#                 virtual_pipeline_model_parallel_size=None,
#                 pipeline_model_parallel_split_rank=None,
#                 use_sharp=False,
#                 context_parallel_size=1,
#                 expert_model_parallel_size=1,
#                 nccl_communicator_config_path=None,
#                 distributed_timeout_minutes=30,
#             )

#             teleaiConfig = _TeleaiParams()
#             torch.manual_seed(1234)
#             model = TeleaiModel(**asdict(teleaiConfig)).cuda(cuda_rank).to(torch.bfloat16)

#             __cfg = __Mock(spec=__TransformerConfig)
#             __cfg._cpu_offloading_context = None
#             __cfg.perform_initialization = True
#             __cfg.use_cpu_initialization = True
#             __cfg.params_dtype = torch.bfloat16
#             __cfg.gradient_accumulation_fusion = False
#             __cfg.expert_model_parallel_size = 1
#             __cfg.defer_embedding_wgrad_compute = False
#             __cfg.async_tensor_model_parallel_allreduce = False
#             __cfg.num_layers = __args.num_layers
#             __cfg.sequence_parallel = False

#             torch.manual_seed(1234)
#             parallel_teleai_model = ParallelTeleaiModel(__cfg).cuda(cuda_rank).to(torch.bfloat16)
#             def __tp_load_state_dict(base_model):
#                 base_dict = base_model.state_dict()
#                 tp_dict = {}
#                 col_w = ["self_attn.query.weight", "self_attn.key.weight", "self_attn.value.weight","ffn.0.weight",
#                          "cross_attn.query.weight", "cross_attn.key.weight", "cross_attn.value.weight",
#                          "cross_attn.img_key.weight", "cross_attn.img_value.weight"]
#                 col_b = ["self_attn.query.bias", "self_attn.key.bias", "self_attn.value.bias","ffn.0.bias",
#                          "cross_attn.query.bias", "cross_attn.key.bias", "cross_attn.value.bias",
#                          "cross_attn.img_key.bias", "cross_attn.img_value.bias"]
#                 row_w = ["ffn.2.weight", "self_attn.out_proj.weight",
#                          "cross_attn.out_proj.weight"]
#                 norm_w = ["self_attn.norm_query.weight", "self_attn.norm_key.weight",
#                           "cross_attn.norm_query.weight", "cross_attn.norm_key.weight",
#                           "cross_attn.norm_image_key.weight"]
#                 def tp_col_weight_load(tp_dict, name, param):
#                     r = mpu.get_tensor_model_parallel_rank()
#                     s = mpu.get_tensor_model_parallel_world_size()
#                     size = param.shape[0] // s
#                     tp_dict[name] = param[r*size:(r+1)*size,:]
#                 def tp_col_bias_load(tp_dict, name, param):
#                     r = mpu.get_tensor_model_parallel_rank()
#                     s = mpu.get_tensor_model_parallel_world_size()
#                     size = param.shape[0] // s
#                     tp_dict[name] = param[r*size:(r+1)*size]
#                 def tp_row_weight_load(tp_dict, name, param):
#                     r = mpu.get_tensor_model_parallel_rank()
#                     s = mpu.get_tensor_model_parallel_world_size()
#                     size = param.shape[1] // s
#                     tp_dict[name] = param[:, r*size:(r+1)*size]
#                 def tp_norm_weight_load(tp_dict, name, param):
#                     r = mpu.get_tensor_model_parallel_rank()
#                     s = mpu.get_tensor_model_parallel_world_size()
#                     size = param.shape[0] // s
#                     tp_dict[name] = param[r*size:(r+1)*size]
#                 for name, param in base_dict.items():
#                     if any(cw in name for cw in col_w):
#                         tp_col_weight_load(tp_dict, name, param)
#                     elif any(cb in name for cb in col_b):
#                         tp_col_bias_load(tp_dict, name, param)
#                     elif any(rw in name for rw in row_w):
#                         tp_row_weight_load(tp_dict, name, param)
#                     elif any(nw in name for nw in norm_w):
#                         tp_norm_weight_load(tp_dict, name, param)
#                     else:
#                         tp_dict[name] = param
#                 return tp_dict
#             parallel_teleai_model.load_state_dict(__tp_load_state_dict(model))
#             input_dict = torch.load("/nvfile-heatstorage/ai_infra/data/lit117/teletron-testing/test_data/saved_inputs_360/input_dict_iter0_rank0.pt", map_location=f"cuda:{cuda_rank}")
#             _ = model(x=input_dict['noisy_latents'],
#                       timestep=input_dict['timestep'],
#                       context=input_dict['prompt_emb']['context'],
#                       clip_feature = input_dict['image_emb']['clip_feature'],
#                       y=input_dict['image_emb']['y'])
#             _ = parallel_teleai_model(x=input_dict['noisy_latents'],
#                       timestep=input_dict['timestep'],
#                       context=input_dict['prompt_emb']['context'],
#                       clip_feature = input_dict['image_emb']['clip_feature'],
#                       y=input_dict['image_emb']['y'])

#             base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#             param_names = {param: name for name, param in model.named_parameters()}
#             timers = NoopTimer()
#             ds_optim = DeepSpeedZeroOptimizer(
#                 base_optimizer,
#                 param_names,
#                 timers=timers,
#                 static_loss_scale=1.0,
#                 dynamic_loss_scale=False,
#                 dynamic_loss_args=None,
#                 clip_grad=0.0,
#                 contiguous_gradients=True,
#                 reduce_bucket_size=500000000,
#                 use_multi_rank_bucket_allreduce=True,
#                 allgather_bucket_size=500000000,
#                 dp_process_group=mpu.get_data_parallel_group(with_context_parallel=True),
#                 expert_parallel_group=None,
#                 expert_data_parallel_group=None,
#                 reduce_scatter=True,
#                 overlap_comm=False,
#                 offload_optimizer_config=None,
#                 mpu=None,
#                 postscale_gradients=True,
#                 gradient_predivide_factor=1.0,
#                 gradient_accumulation_steps=1,
#                 ignore_unused_parameters=True,
#                 partition_grads=True,
#                 round_robin_gradients=False,
#                 has_moe_layers=False,
#                 fp16_master_weights_and_gradients=False,
#                 elastic_checkpoint=False,
#             )

#         # Prepare checkpoint files in DS dir: latest, model_optim_rng.pt, zero2 shards.
#         model_ckpt_path = get_checkpoint_name(load_dir, iteration, release=False, return_base_dir=False)
        
#         # Mock 64 shards for testing parallel load
#         with __patch("megatron.core.mpu.get_data_parallel_world_size", return_value=64):
#             optim_paths = get_checkpoint_name(load_dir, iteration, release=False, return_base_dir=False, use_zero2=True)
            
#         if rank == 0:
#             tracker = get_checkpoint_tracker_filename(load_dir)
#             os.makedirs(load_dir, exist_ok=True)
#             parent_dir = os.path.dirname(model_ckpt_path)
#             if parent_dir:
#                 os.makedirs(parent_dir, exist_ok=True)
#             with open(tracker, "w") as f:
#                 f.write(str(iteration))
#             import random, numpy as np
#             random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed(1234)
#             rng_entry = {
#                 "random_rng_state": random.getstate(),
#                 "np_rng_state": np.random.get_state(),
#                 "torch_rng_state": torch.get_rng_state(),
#                 "cuda_rng_state": torch.cuda.get_rng_state(),
#                 "rng_tracker_states": {"model-parallel-rng": [0]},
#             }
#             torch.save(
#                 {
#                     "iteration": iteration,
#                     "args": types.SimpleNamespace(consumed_train_samples=0, consumed_valid_samples=0, dit_world_size=world_size),
#                     "model": model.state_dict(),
#                     "opt_param_scheduler": {"num_steps": 1},
#                 "rng_state": [rng_entry for _ in range(world_size)],
#             },
#             model_ckpt_path,
#             )
#             # Save 64 dummy shards
#             for pth in optim_paths:
#                 os.makedirs(os.path.dirname(pth), exist_ok=True)
#                 # Just save a small dict to be fast, deepspeed state dict format
#                 # We reuse ds_optim.state_dict() but it might be large.
#                 # Let's save a dummy dict that looks like a shard.
#                 # But if we want real load, maybe we should save real state?
#                 # ds_optim.state_dict() returns the whole state or local shard?
#                 # DeepSpeed Zero2 state_dict() usually returns local shard.
#                 # To simulate 64 shards, we just copy the same content 64 times.
#                 torch.save(ds_optim.state_dict(), pth)
#             q.put(f"deepspeed_model_and_zero2_shards_written rank{rank} count={len(optim_paths)}")
#         try:
#             torch.distributed.barrier(device_ids=[cuda_rank])
#         except TypeError:
#             torch.distributed.barrier()

#         model_ckpt_path = get_checkpoint_name(load_dir, iteration, release=False, return_base_dir=False)
#         if rank == 0:
#             tracker = get_checkpoint_tracker_filename(load_dir)
#             os.makedirs(load_dir, exist_ok=True)
#             parent_dir = os.path.dirname(model_ckpt_path)
#             if parent_dir:
#                 os.makedirs(parent_dir, exist_ok=True)
#             with open(tracker, "w") as f:
#                 f.write(str(iteration))
#             random.seed(1234)
#             np.random.seed(1234)
#             torch.manual_seed(1234)
#             torch.cuda.manual_seed(1234)
#             rng_entry = {
#                 "random_rng_state": random.getstate(),
#                 "np_rng_state": np.random.get_state(),
#                 "torch_rng_state": torch.get_rng_state(),
#                 "cuda_rng_state": torch.cuda.get_rng_state(),
#                 "rng_tracker_states": {"model-parallel-rng": [0]},
#             }
#             ckpt_args = types.SimpleNamespace(
#                 consumed_train_samples=0,
#                 consumed_valid_samples=0,
#                 dit_world_size=world_size,
#             )
#             torch.save(
#                 {
#                     "iteration": iteration,
#                     "args": ckpt_args,
#                     "model": model.state_dict(),
#                     "opt_param_scheduler": {"num_steps": 1},
#                     "rng_state": [rng_entry for _ in range(world_size)],
#                 },
#                 model_ckpt_path,
#             )
#             q.put(f"deepspeed_model_ckpt_written rank{rank} path={os.path.basename(model_ckpt_path)}")

#         try:
#             torch.distributed.barrier(device_ids=[cuda_rank])
#         except TypeError:
#             torch.distributed.barrier()

#         runtime_args = types.SimpleNamespace(
#             load=load_dir,
#             use_zero2=True,
#             no_load_optim=False,
#             no_load_rng=False,
#             finetune=False,
#             consumed_train_samples=0,
#             consumed_valid_samples=0,
#             auto_detect_ckpt_format=False,
#             use_dist_ckpt=False,
#             data_parallel_random_init=True,
#             retro_add_retriever=False,
#             lora=False,
#             fp16=False,
#             bf16=False,
#             pretrained_checkpoint=None,
#             exit_on_missing_checkpoint=False,
#             use_distributed_optimizer=False,
#             dit_world_size=world_size,
#         )

#         mixin = CheckPointMixin()

#         with _patch("teletron.train.checkpoint.checkpoint.get_args", return_value=runtime_args), _patch(
#             "teletron.train.checkpoint.checkpoint.update_num_microbatches", return_value=None
#         ), _patch(
#             "teletron.train.checkpoint.checkpoint.sys.exit",
#             side_effect=RuntimeError("sys.exit called during load_checkpoint"),
#         ), _patch("megatron.core.mpu.get_data_parallel_world_size", return_value=64):
#             # torch.distributed.breakpoint()
#             it, _flops, _opt, _sched = mixin.load_checkpoint([model], ds_optim, None)
#         try:
#             device_ok = False
#             found = False
#             for attr in ("fp32_partitioned_groups_flat", "fp16_groups_flat"):
#                 t = getattr(ds_optim, attr, None)
#                 if isinstance(t, list) and len(t) > 0 and torch.is_tensor(t[0]):
#                     found = True
#                     device_ok = t[0].is_cuda
#                     break
#             q.put(f"deepspeed_optimizer_cuda rank{rank} cuda_ok={device_ok}")
#             if found and not device_ok:
#                 raise RuntimeError("deepspeed optimizer tensors not on cuda")
#         except Exception as e:
#             q.put(f"deepspeed_optimizer_cuda_check_exception rank{rank} error={str(e)}")
#         q.put(f"deepspeed_load_checkpoint_ok rank{rank} iter={it}")
#     except Exception as e:
#         q.put(
#             f"deepspeed_load_checkpoint_exception rank{rank} cuda={cuda_rank} "
#             f"world_size={world_size} error={str(e)}"
#         )
#         import traceback

#         traceback.print_exc()
#     finally:
#         try:
#             if torch.distributed.is_initialized():
#                 torch.distributed.destroy_process_group()
#         except Exception:
#             pass


# def _parallel_zero2_load(rank, world_size, q, tp_size, cp_size):
#     from unittest.mock import Mock
#     import datetime
#     with patch("teletron.utils.get_args") as mock_get_args:
#         args = Mock()
#         args.distributed_vae = False
#         args.consumer_models_num = 1
#         args.distributed_vae_world_size = 0
#         mock_get_args.return_value = args
#     from teletron.core.parallel_state import initialize_model_parallel_base
#     # Prefer a resilient device mapping: default to 0,1,2,3 and fall back to modulo mapping
#     if "CUDA_VISIBLE_DEVICES" not in os.environ:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#     if not torch.cuda.is_available():
#         q.put(f"cuda_unavailable rank{rank}")
#         return
#     device_count = torch.cuda.device_count()
#     if device_count == 0:
#         q.put(f"no_cuda_devices_visible rank{rank}")
#         return
#     cuda_devices = list(range(device_count))
#     cuda_rank = cuda_devices[rank % device_count]
#     torch.cuda.set_device(cuda_rank)
#     q.put(f"worker_start_zero2 rank{rank} cuda={cuda_rank}")
#     try:
#         torch.distributed.init_process_group(
#             backend="nccl",
#             init_method="env://",
#             world_size=world_size,
#             rank=rank,
#             timeout=datetime.timedelta(minutes=3),
#         )
#         q.put(f"zero2_pg_ready rank{rank} world_size={world_size}")
#         with patch("teletron.utils.get_args", return_value=args):
#             initialize_model_parallel_base(
#                 tensor_model_parallel_size=tp_size,
#                 pipeline_model_parallel_size=1,
#                 virtual_pipeline_model_parallel_size=None,
#                 pipeline_model_parallel_split_rank=None,
#                 use_sharp=False,
#                 context_parallel_size=cp_size,
#                 expert_model_parallel_size=1,
#                 nccl_communicator_config_path=None,
#                 distributed_timeout_minutes=30,
#             )
#         try:
#             q.put(f"begin_sanity rank{rank}")
#             sanity = torch.ones(1, device=f"cuda:{cuda_rank}") * (rank + 1)
#             torch.distributed.all_reduce(sanity, op=torch.distributed.ReduceOp.SUM)
#             expected = sum(range(1, world_size + 1))
#             if int(sanity.item()) != expected:
#                 q.put(f"sanity_allreduce_fail rank{rank}")
#                 return
#             q.put(f"sanity_allreduce_success rank{rank}")
#         except Exception as e:
#             q.put(f"sanity_allreduce_exception rank{rank} error={str(e)}")
#             return

#         base_dir = os.environ.get("TELETRON_OPTIM_DIR", "")
#         cp_world = mpu.get_context_parallel_world_size()

#         pattern = os.path.join(base_dir, "zero2_optim_*.pt")
#         all_files = sorted(glob.glob(pattern))

#         desired_dp_count = int(os.environ.get("DESIRED_DP_COUNT", "8"))
#         target_count = desired_dp_count * cp_world
#         q.put(f"zero2_files_found rank{rank} total_found={len(all_files)} target_count={target_count} cp_world={cp_world}")

#         if rank == 0:
#             print(f"[Init] Found {len(all_files)} weight files in {base_dir}. Target count: {target_count}")
#             if len(all_files) > 0:
#                 print(f"[Init] Sample files: {[os.path.basename(f) for f in all_files[:3]]}")

#         if len(all_files) >= target_count:
#             file_paths = all_files[:target_count]
#         else:
#             file_paths = all_files
#             if rank == 0:
#                 print(f"[Init] WARNING: Found fewer files ({len(file_paths)}) than expected ({target_count})")

#         q.put(f"debug_rank{rank}_files_count={len(file_paths)}")

#         mixin = CheckPointMixin()
#         state_dict = {}
#         t0 = time.time()

#         q.put(f"debug_rank{rank}_start_load")
#         try:
#             mixin.load_zero2_optimizer(file_paths, state_dict)
#             q.put(f"debug_rank{rank}_end_load_success")
#         except Exception as e:
#             q.put(f"debug_rank{rank}_load_failed error={str(e)}")
#             raise

#         t1 = time.time()

#         total_files = len(file_paths)
#         start_idx = (total_files * rank) // world_size
#         end_idx = (total_files * (rank + 1)) // world_size
#         q.put(f"zero2_shard_plan rank{rank} start={start_idx} end={end_idx} total={total_files} world_size={world_size}")
#         q.put(
#             f"zero2_rank_files rank{rank} files="
#             f"{[os.path.basename(file_paths[i]) for i in range(start_idx, end_idx)]}"
#         )

#         loaded_indices = []
#         failed_indices = []

#         for i in range(start_idx, end_idx):
#             if i < len(state_dict["optimizer"]) and state_dict["optimizer"][i] is not None:
#                 loaded_indices.append(i)
#             else:
#                 failed_indices.append(i)

#         unexpected_indices = []
#         for i in range(total_files):
#             if (i < start_idx or i >= end_idx) and i < len(state_dict["optimizer"]):
#                 if state_dict["optimizer"][i] is not None:
#                     unexpected_indices.append(i)

#         if len(failed_indices) == 0:
#             loaded_names = [os.path.basename(file_paths[i]) for i in loaded_indices]
#             size_bytes = sum(os.path.getsize(file_paths[i]) for i in loaded_indices)
#             q.put(f"zero2_shards_loaded rank{rank} loaded_count={len(loaded_indices)} loaded_size={size_bytes}B")
#             q.put(
#                 f"zero2_load_16_success rank{rank} count={len(loaded_indices)} files={loaded_names} "
#                 f"size={size_bytes}B duration={t1-t0:.3f}s"
#             )
#             if len(unexpected_indices) > 0:
#                 q.put(f"zero2_load_16_warning rank{rank} unexpected_loaded={unexpected_indices}")
#         else:
#             q.put(f"zero2_load_16_missing_files rank{rank} missing={failed_indices} unexpected={unexpected_indices}")
#     except Exception as e:
#         q.put(f"zero2_load_16_exception rank{rank} {str(e)}")
#         return
#     finally:
#         try:
#             if torch.distributed.is_initialized():
#                 torch.distributed.destroy_process_group()
#         except Exception:
#             pass

# def _spawn(world_size, fn, *args):
#     def get_free_port():
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("127.0.0.1", 0))
#         port = s.getsockname()[1]
#         s.close()
#         return port
#     os.environ["WORLD_SIZE"] = str(world_size)
#     os.environ["MASTER_ADDR"] = "127.0.0.1"
#     os.environ["MASTER_PORT"] = str(get_free_port())
#     ctx = mp.get_context("spawn")
#     q = ctx.Queue()
#     processes = []
#     for i in range(world_size):
#         p = ctx.Process(target=fn, args=(i, world_size, q) + args)
#         p.start()
#         processes.append(p)

#     join_timeout_s = float(os.environ.get("SPAWN_JOIN_TIMEOUT_S", "600"))
#     deadline = time.time() + max(join_timeout_s, 1.0)
#     for p in processes:
#         remaining = deadline - time.time()
#         if remaining <= 0:
#             break
#         p.join(remaining)

#     for p in processes:
#         if p.is_alive():
#             try:
#                 p.terminate()
#             except Exception:
#                 pass
#     for p in processes:
#         try:
#             p.join(5)
#         except Exception:
#             pass

#     return q

# def _drain_queue(q, label, total_timeout_s=60, idle_timeout_s=2):
#     res = []
#     start_t = time.time()
#     last_msg_t = start_t
#     while True:
#         now = time.time()
#         if now - start_t > total_timeout_s:
#             break
#         if now - last_msg_t > idle_timeout_s:
#             break
#         try:
#             msg = q.get(timeout=0.5)
#             res.append(msg)
#             print(f"[{label}] {msg}", flush=True)
#             last_msg_t = time.time()
#         except pyqueue.Empty:
#             continue
#     return res


# class TestZero2RealLoad(TestCase):
#     def test_real_dir_load_16(self):
#         os.environ["TELETRON_OPTIM_DIR"] = "/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00"
#         world_size = 4
#         tp = 2
#         cp = 2
#         q = _spawn(world_size, _parallel_zero2_load, tp, cp)
#         res = _drain_queue(q, "MainProcess/zero2", total_timeout_s=120, idle_timeout_s=5)
#         self.assertGreater(len(res), 0, "no worker messages received")
#         sanity = [r for r in res if r.startswith("sanity_allreduce_success")]
#         self.assertEqual(len(sanity), world_size)
#         loads = [r for r in res if r.startswith("zero2_load_16_success")]
#         self.assertEqual(len(loads), world_size)
#         # multi4_ok = [r for r in res if r.startswith("multi_load4_success")]
#         # self.assertEqual(len(multi4_ok), 16)

#     def test_two_card_load_8(self):
#         os.environ["TELETRON_OPTIM_DIR"] = "/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00"
#         os.environ["DESIRED_DP_COUNT"] = "4"
#         world_size = 2
#         tp = 1
#         cp = 2
#         q = _spawn(world_size, _parallel_zero2_load, tp, cp)
#         res = _drain_queue(q, "MainProcess/zero2-2cards", total_timeout_s=120, idle_timeout_s=5)
#         self.assertGreater(len(res), 0, "no worker messages received")
#         sanity = [r for r in res if r.startswith("sanity_allreduce_success")]
#         self.assertEqual(len(sanity), world_size)
#         load_ok = [r for r in res if r.startswith("zero2_load_16_success")]
#         load_miss = [r for r in res if r.startswith("zero2_load_16_missing_files")]
#         self.assertEqual(len(load_ok) + len(load_miss), world_size)
#         # multi4_ok = [r for r in res if r.startswith("multi_load4_success")]
#         # self.assertEqual(len(multi4_ok), 8)





# class TestDeepSpeedZero2LoadCheckpoint(TestCase):
#     def test_deepspeed_zero2_load_checkpoint_end_to_end(self):
#         if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
#             self.skipTest("need >=4 visible cuda devices")
#         try:
#             import deepspeed  # noqa: F401
#         except Exception as e:
#             self.skipTest(f"deepspeed unavailable: {e}")

#         base_dir = tempfile.mkdtemp(prefix="ds_zero2_load_ckpt_")
#         os.environ["DS_ZERO2_LOAD_CKPT_DIR"] = base_dir
#         os.environ["DS_ZERO2_LOAD_CKPT_ITER"] = "1"
        
#         world_size = 4
#         q = _spawn(world_size, _parallel_deepspeed_zero2_load_checkpoint)
#         res = _drain_queue(q, "MainProcess/ds_zero2_load_ckpt", total_timeout_s=600, idle_timeout_s=20)
#         self.assertGreater(len(res), 0, "no worker messages received")
#         unavailable = [r for r in res if r.startswith("deepspeed_unavailable")]
#         if len(unavailable) > 0:
#             self.skipTest(unavailable[0])
#         ok = [r for r in res if r.startswith("deepspeed_load_checkpoint_ok")]
#         self.assertEqual(len(ok), world_size)

# # def _parallel_full_load(rank, world_size, q, tp_size, cp_size):
# #     from unittest.mock import MagicMock, patch
# #     import numpy as np
# #     import random
# #     import datetime

# #     if "CUDA_VISIBLE_DEVICES" not in os.environ:
# #         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# #     if not torch.cuda.is_available():
# #         q.put(f"cuda_unavailable rank{rank}")
# #         return
# #     device_count = torch.cuda.device_count()
# #     if device_count == 0:
# #         q.put(f"no_cuda_devices_visible rank{rank}")
# #         return

# #     cuda_rank = 0 if world_size == 1 else (rank % device_count)
# #     torch.cuda.set_device(cuda_rank)
# #     q.put(f"worker_start_full_load rank{rank} cuda={cuda_rank}")

# #     def _cleanup_dist():
# #         try:
# #             if torch.distributed.is_initialized():
# #                 torch.distributed.destroy_process_group()
# #         except Exception:
# #             pass

# #     try:
# #         torch.distributed.init_process_group(
# #             backend="nccl",
# #             init_method="env://",
# #             world_size=world_size,
# #             rank=rank,
# #             timeout=datetime.timedelta(minutes=3),
# #         )
# #         q.put(f"full_load_pg_ready rank{rank} world_size={world_size}")
# #         sanity = torch.ones(1, device=f"cuda:{cuda_rank}") * (rank + 1)
# #         torch.distributed.all_reduce(sanity, op=torch.distributed.ReduceOp.SUM)
# #         expected = sum(range(1, world_size + 1))
# #         if int(sanity.item()) != expected:
# #             q.put(f"full_load_sanity_allreduce_fail rank{rank}")
# #             _cleanup_dist()
# #             return
# #         q.put(f"full_load_sanity_allreduce_success rank{rank}")
# #     except Exception as e:
# #         q.put(f"full_load_dist_init_exception rank{rank} error={str(e)}")
# #         _cleanup_dist()
# #         return

# #     # Prepare Mock Args
# #     mock_args = MagicMock()
# #     # Required args for load_checkpoint
# #     mock_args.load = "dummy_load_path"
# #     mock_args.use_zero2 = True
# #     mock_args.no_load_optim = False
# #     mock_args.no_load_rng = False
# #     mock_args.finetune = False
# #     mock_args.consumed_train_samples = 0
# #     mock_args.consumed_valid_samples = 0
# #     mock_args.auto_detect_ckpt_format = False
# #     mock_args.use_dist_ckpt = False
# #     mock_args.data_parallel_random_init = True
# #     mock_args.retro_add_retriever = False
# #     mock_args.lora = False
# #     mock_args.fp16 = False
# #     mock_args.bf16 = True
# #     mock_args.distributed_vae = False
# #     mock_args.consumer_models_num = 1
# #     mock_args.distributed_vae_world_size = 0
# #     mock_args.pretrained_checkpoint = None
# #     mock_args.exit_on_missing_checkpoint = False
# #     mock_args.use_distributed_optimizer = False
# #     mock_args.no_save_optim = False
# #     mock_args.no_save_rng = False
# #     mock_args.dit_world_size = world_size # Assuming this is used for scheduler scaling

# #     # Identify Real Files
# #     base_dir = os.environ.get("TELETRON_OPTIM_DIR", "")
# #     pattern = os.path.join(base_dir, "zero2_optim_*.pt")
# #     all_optim_files = sorted(glob.glob(pattern))
    
# #     # Filter files if DESIRED_DP_COUNT is set (optional, consistent with previous test)
# #     desired_dp_count = int(os.environ.get("DESIRED_DP_COUNT", "8"))
# #     target_count = desired_dp_count * max(int(cp_size), 1)
# #     if len(all_optim_files) >= target_count:
# #         all_optim_files = all_optim_files[:target_count]
# #     q.put(
# #         f"full_load_files rank{rank} found={len(all_optim_files)} desired_dp={desired_dp_count} "
# #         f"cp_size={cp_size} target_count={target_count} world_size_sim={world_size}"
# #     )
# #     if rank == 0 and world_size > 1:
# #         for r in range(world_size):
# #             s = (len(all_optim_files) * r) // world_size
# #             e = (len(all_optim_files) * (r + 1)) // world_size
# #             q.put(f"full_load_rank_files rank{r} start={s} end={e} files={[os.path.basename(p) for p in all_optim_files[s:e]]}")
    
# #     if rank == 0:
# #         print(f"[FullLoad] Rank {rank} targeting {len(all_optim_files)} optimizer files from {base_dir}")
# #         if len(all_optim_files) > 0:
# #             print(f"[FullLoad] Sample optim files: {[os.path.basename(p) for p in all_optim_files[:3]]}")

# #     # Prepare Mocks for Checkpoint Loading
# #     model = [MagicMock()]
# #     model[0].sharded_state_dict.return_value = {}
# #     model[0].state_dict_for_save_checkpoint.return_value = {}
# #     model_called = {"flag": False}
# #     model[0].load_state_dict.side_effect = lambda *_args, **_kwargs: model_called.__setitem__("flag", True)

# #     optimizer = _RealZero2Optimizer(rank=rank, q=q)
    
# #     scheduler = MagicMock()
    
# #     # Setup CheckPointMixin
# #     mixin = CheckPointMixin()
    
# #     # Prepare Patch Context
# #     # We need to patch:
# #     # 1. get_args -> mock_args
# #     # 2. read_metadata -> (iteration, release)
# #     # 3. get_checkpoint_name -> returns real paths for zero2 files, dummy for model
# #     # 4. torch.load -> intercept model load, allow optimizer load
# #     # 5. dist_checkpointing.check_is_distributed_checkpoint -> False
# #     # 6. checkpoint_exists -> True (to bypass initial check)
    
# #     original_torch_load = torch.load

# #     with patch("teletron.train.checkpoint.checkpoint.get_args", return_value=mock_args), \
# #          patch("teletron.train.checkpoint.checkpoint.read_metadata", return_value=(1000, False)), \
# #          patch("teletron.train.checkpoint.checkpoint.dist_checkpointing.check_is_distributed_checkpoint", return_value=False), \
# #          patch("teletron.train.checkpoint.checkpoint.checkpoint_exists", return_value=True), \
# #          patch("teletron.train.checkpoint.checkpoint.update_num_microbatches", return_value=None), \
# #          patch("teletron.train.checkpoint.checkpoint.sys.exit", side_effect=RuntimeError("sys.exit called during load_checkpoint")), \
# #          patch("teletron.train.checkpoint.checkpoint.get_checkpoint_name") as mock_get_ckpt_name, \
# #          patch("teletron.train.checkpoint.checkpoint.torch.load") as mock_torch_load, \
# #          patch.object(CheckPointMixin, "_load_zero2_checkpoint") as mock_zero2_loader, \
# #          patch("teletron.train.checkpoint.utils.get_args", return_value=mock_args), \
# #          patch("megatron.core.mpu.get_data_parallel_rank", return_value=rank), \
# #          patch("megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0), \
# #          patch("megatron.core.mpu.get_pipeline_model_parallel_rank", return_value=0), \
# #          patch("random.setstate") as mock_random_setstate, \
# #          patch("numpy.random.set_state") as mock_np_set_state, \
# #          patch("torch.set_rng_state") as mock_torch_set_rng_state, \
# #          patch("torch.cuda.set_rng_state") as mock_cuda_set_rng_state, \
# #          patch("teletron.train.checkpoint.checkpoint.tensor_parallel") as mock_tp:

# #         # Mock get_checkpoint_name
# #         def get_ckpt_name_side_effect(load_dir, iteration, release=False, return_base_dir=False, use_zero2=False, **kwargs):
# #             if use_zero2:
# #                 # Must return the list of real file paths
# #                 return all_optim_files
# #             if return_base_dir:
# #                 return "dummy_base_dir"
# #             return "dummy_model_optim_rng.pt"
# #         mock_get_ckpt_name.side_effect = get_ckpt_name_side_effect

# #         # Mock torch.load
# #         def torch_load_side_effect(f, map_location=None, weights_only=False):
# #             # Check if f is one of our real optimizer files
# #             # Note: f could be a Path object or string
# #             f_str = str(f)
# #             if "zero2_optim_" in f_str and os.path.exists(f_str):
# #                 return original_torch_load(f, map_location=map_location, weights_only=weights_only)
            
# #             # Otherwise return dummy state dict for model/rng
# #             # Construct a dummy state dict that load_checkpoint expects
# #             dummy_state = {
# #                 "iteration": 1000,
# #                 "args": mock_args,
# #                 "model": {}, # model load will use this
# #                 "optimizer": [], # Will be ignored/overwritten by load_zero2_optimizer logic if use_zero2 is True? 
# #                                  # No, load_zero2_optimizer modifies this dict in-place.
# #                                  # But wait, load_checkpoint calls _load_zero2_checkpoint, which returns state_dict.
# #                                  # _load_zero2_checkpoint loads model_checkpoint_name first.
# #                                  # So this dictionary is what _load_zero2_checkpoint gets from torch.load(model_checkpoint_name)
# #                 "rng_state": [ # rng_state list for data parallel ranks
# #                      {
# #                         'random_rng_state': random.getstate(),
# #                         'np_rng_state': np.random.get_state(),
# #                         'torch_rng_state': torch.get_rng_state(),
# #                         'cuda_rng_state': torch.cuda.get_rng_state(),
# #                         'rng_tracker_states': {'model-parallel-rng': [0]} 
# #                      } for _ in range(world_size) # Assuming enough for DP
# #                 ],
# #                 "opt_param_scheduler": {"num_steps": 1},
# #             }
# #             return dummy_state
# #         mock_torch_load.side_effect = torch_load_side_effect
# #         # Patch _load_zero2_checkpoint to ensure Zero2 path always loads real optimizer shards
# #         def fake_zero2_loader(load_dir, checkpoint_step=None):
# #             state = torch_load_side_effect("dummy_model_optim_rng.pt")
# #             total_files = len(all_optim_files)
# #             start_idx = (total_files * rank) // world_size
# #             end_idx = (total_files * (rank + 1)) // world_size
# #             optimizer_state_list = [None] * total_files
# #             shard_paths = [all_optim_files[i] for i in range(start_idx, end_idx)]
# #             per_rank_procs = int(os.environ.get("ZERO2_SUBPROC_PER_RANK", "16"))
# #             procs = min(max(per_rank_procs, 1), max(len(shard_paths), 1))
# #             q.put(f"full_load_zero2_subproc rank{rank} procs={procs} tasks={len(shard_paths)}")
# #             q.put(f"full_load_zero2_map_location rank{rank} mode=cpu_then_cuda device=cuda:{cuda_rank}")
# #             ctx = mp.get_context("spawn")
# #             with ctx.Pool(processes=procs) as pool:
# #                 loaded_shards_cpu = pool.map(_load_zero2_file_cpu, shard_paths)
# #             loaded_shards = [_move_to_cuda(sd, cuda_rank) for sd in loaded_shards_cpu]
# #             for off, i in enumerate(range(start_idx, end_idx)):
# #                 optimizer_state_list[i] = loaded_shards[off]
# #             state["optimizer"] = optimizer_state_list
# #             size_bytes = 0
# #             for i in range(start_idx, end_idx):
# #                 try:
# #                     size_bytes += os.path.getsize(all_optim_files[i])
# #                 except Exception:
# #                     pass
# #             q.put(
# #                 f"full_load_zero2_shards rank{rank} start={start_idx} end={end_idx} total={total_files} "
# #                 f"loaded_count={max(end_idx-start_idx,0)} loaded_size={size_bytes}B"
# #             )
# #             q.put(
# #                 f"full_load_zero2_rank_files rank{rank} files="
# #                 f"{[os.path.basename(all_optim_files[i]) for i in range(start_idx, end_idx)]}"
# #             )
# #             return state, "dummy_base_dir", False
# #         mock_zero2_loader.side_effect = fake_zero2_loader
# #         rng_states_called = {"flag": False}
# #         mock_tp.get_cuda_rng_tracker.return_value.set_states.side_effect = lambda _s: rng_states_called.__setitem__("flag", True)
# #         try:
# #             iteration, num_flops, opt, sched = mixin.load_checkpoint(model, optimizer, scheduler)
# #             q.put(f"full_load_return rank{rank} iter={iteration}")
# #             if model_called["flag"]:
# #                 q.put(f"model_state_dict_applied rank{rank}")
# #             else:
# #                 q.put(f"model_state_dict_missing rank{rank}")
# #             if optimizer.called:
# #                 q.put(f"optimizer_load_state_dict_called rank{rank}")
# #             else:
# #                 q.put(f"optimizer_load_state_dict_missing rank{rank}")
# #             q.put(f"rng_basic_applied rank{rank} random={mock_random_setstate.called} numpy={mock_np_set_state.called} torch={mock_torch_set_rng_state.called} cuda={mock_cuda_set_rng_state.called}")
# #             if rng_states_called["flag"]:
# #                 q.put(f"rng_states_applied rank{rank}")
# #             else:
# #                 q.put(f"rng_states_missing rank{rank}")
# #             q.put(f"full_load_ok rank{rank}")
# #         except Exception as e:
# #             q.put(f"full_load_exception rank{rank} error={str(e)}")
# #             import traceback
# #             traceback.print_exc()
# #             _cleanup_dist()
# #             return

# #     gc.collect()
# #     try:
# #         torch.cuda.empty_cache()
# #     except Exception:
# #         pass
# #     finally:
# #         try:
# #             if torch.distributed.is_initialized():
# #                 torch.distributed.destroy_process_group()
# #         except Exception:
# #             pass

# # class TestFullLoad(TestCase):
# #     def test_full_load_checkpoint_flow(self):
# #         os.environ["TELETRON_OPTIM_DIR"] = "/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00"
# #         os.environ["DESIRED_DP_COUNT"] = "64"
# #         os.environ["ZERO2_SUBPROC_PER_RANK"] = "16"
# #         if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
# #             self.skipTest("cuda unavailable")
# #         world_size = 4
# #         tp = 1
# #         cp = 1
# #         q = _spawn(world_size, _parallel_full_load, tp, cp)
# #         res = _drain_queue(q, "MainProcess/full_load", total_timeout_s=180, idle_timeout_s=8)
# #         self.assertGreater(len(res), 0, "no worker messages received")
# #         ok = [r for r in res if r.startswith("full_load_ok")]
# #         self.assertEqual(len(ok), world_size)
# #         opt_called = [r for r in res if r.startswith("optimizer_load_state_dict_called")]
# #         self.assertEqual(len(opt_called), world_size)
# #         model_called = [r for r in res if r.startswith("model_state_dict_applied")]
# #         self.assertEqual(len(model_called), world_size)
# #         shard_msgs = [r for r in res if r.startswith("full_load_zero2_shards")]
# #         self.assertEqual(len(shard_msgs), world_size)
# #         per_rank_files = [r for r in res if r.startswith("full_load_zero2_rank_files")]
# #         self.assertEqual(len(per_rank_files), world_size)
# #         subproc_msgs = [r for r in res if r.startswith("full_load_zero2_subproc")]
# #         self.assertEqual(len(subproc_msgs), world_size)
# #         map_loc_msgs = [r for r in res if r.startswith("full_load_zero2_map_location")]
# #         self.assertEqual(len(map_loc_msgs), world_size)
# #         self.assertTrue(all("device=cuda:" in m for m in map_loc_msgs))
# #         opt_dev_msgs = [r for r in res if r.startswith("optimizer_real_first_tensor_device")]
# #         self.assertEqual(len(opt_dev_msgs), world_size)
# #         self.assertTrue(all("device=cuda:" in m for m in opt_dev_msgs))
# #         shards = []
# #         totals = set()
# #         expected_total = int(os.environ.get("DESIRED_DP_COUNT", "8")) * cp
# #         for msg in shard_msgs:
# #             parts = msg.split()
# #             rank_part = parts[1]
# #             rank = int(rank_part.replace("rank", ""))
# #             kv = {}
# #             for p in parts[2:]:
# #                 if "=" in p:
# #                     k, v = p.split("=", 1)
# #                     kv[k] = v
# #             start = int(kv["start"])
# #             end = int(kv["end"])
# #             total = int(kv["total"])
# #             loaded_count = int(kv["loaded_count"])
# #             totals.add(total)
# #             shards.append((rank, start, end, loaded_count))
# #         self.assertEqual(totals, {expected_total})
# #         shards_sorted = sorted(shards, key=lambda x: x[1])
# #         self.assertEqual(shards_sorted[0][1], 0)
# #         for i, (_rank, start, end, loaded_count) in enumerate(shards_sorted):
# #             self.assertEqual(end - start, loaded_count)
# #             if i + 1 < len(shards_sorted):
# #                 self.assertEqual(end, shards_sorted[i + 1][1])
# #         self.assertEqual(shards_sorted[-1][2], expected_total)
import os
import socket
import time
import glob
import torch
import gc
import queue as pyqueue
import multiprocessing as mp
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import TestCase
from unittest.mock import patch
from unit_tests.test_utils import spawn
from megatron.core import mpu
from teletron.train.checkpoint.checkpoint import CheckPointMixin


def _load_zero2_file_gpu(args):
    import torch as _torch
    path, device = args
    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in zero2 shard loader subprocess")
    _torch.cuda.set_device(device)
    return _torch.load(path, map_location=f"cuda:{device}", weights_only=False)


def _load_zero2_file_cpu(path: str):
    import torch as _torch
    return _torch.load(path, map_location="cpu", weights_only=False)


def _move_to_cuda(obj, device: int):
    if torch.is_tensor(obj):
        return obj.to(device=f"cuda:{device}", non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cuda(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cuda(v, device) for v in obj)
    return obj


def _find_first_tensor_device(obj):
    if torch.is_tensor(obj):
        return str(obj.device)
    if isinstance(obj, dict):
        for v in obj.values():
            d = _find_first_tensor_device(v)
            if d is not None:
                return d
    if isinstance(obj, list) or isinstance(obj, tuple):
        for v in obj:
            d = _find_first_tensor_device(v)
            if d is not None:
                return d
    return None


class _RealZero2Optimizer:
    def __init__(self, rank: int, q):
        self.rank = rank
        self.q = q
        self.called = False
        self.loaded_shard_indices = []
        self.loaded_shard_keys = {}

    def load_state_dict(self, state_dict, load_from_fp32_weights=False):
        self.called = True
        if not isinstance(state_dict, list):
            raise TypeError(f"expected list for zero2 optimizer state, got {type(state_dict)}")

        loaded = []
        keys_by_idx = {}
        for i, shard in enumerate(state_dict):
            if shard is None:
                continue
            if not isinstance(shard, dict):
                raise TypeError(f"expected shard dict at index {i}, got {type(shard)}")
            loaded.append(i)
            try:
                keys_by_idx[i] = sorted(list(shard.keys()))[:20]
            except Exception:
                keys_by_idx[i] = []
        self.loaded_shard_indices = loaded
        self.loaded_shard_keys = keys_by_idx
        first_device = _find_first_tensor_device(state_dict)
        if first_device is not None:
            self.q.put(f"optimizer_real_first_tensor_device rank{self.rank} device={first_device}")
        self.q.put(
            f"optimizer_real_loaded rank{self.rank} loaded_shards={len(loaded)} "
            f"load_from_fp32_weights={bool(load_from_fp32_weights)}"
        )


def _parallel_deepspeed_zero2_load_checkpoint(rank, world_size, q):
    import datetime
    import types
    import random
    import numpy as np
    import torch.nn as nn
    from unittest.mock import patch as _patch

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if not torch.cuda.is_available():
        q.put(f"cuda_unavailable rank{rank}")
        return
    device_count = torch.cuda.device_count()
    if device_count == 0:
        q.put(f"no_cuda_devices_visible rank{rank}")
        return

    cuda_rank = rank % device_count
    torch.cuda.set_device(cuda_rank)

    try:
        import deepspeed
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        from deepspeed.utils.timer import NoopTimer
    except Exception as e:
        q.put(f"deepspeed_unavailable rank{rank} error={str(e)}")
        return

    load_dir = os.environ.get("DS_ZERO2_LOAD_CKPT_DIR", "")
    if not load_dir:
        q.put(f"deepspeed_no_dir rank{rank}")
        return
    iteration = int(os.environ.get("DS_ZERO2_LOAD_CKPT_ITER", "1"))
    
    # Ensure 16 threads per rank for loading
    os.environ["ZERO2_LOAD_THREADS"] = "16"
    os.environ["ZERO2_SUBPROC_PER_RANK"] = "16"

    try:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=3),
        )
        deepspeed.init_distributed()

        from teletron.core.parallel_state import initialize_model_parallel_base
        from teletron.train.checkpoint.utils import (
            get_checkpoint_name,
            get_checkpoint_tracker_filename,
        )

        from unittest.mock import patch as __patch, Mock as __Mock
        from dataclasses import dataclass, asdict
        from typing import Tuple
        from megatron.core.transformer import TransformerConfig as __TransformerConfig
        from megatron.core import mpu as __mpu

        @dataclass
        class _TeleaiParams:
            dim: int = 5120
            in_dim: int = 36
            out_dim: int = 16
            text_dim: int = 4096
            freq_dim: int = 256
            ffn_dim: int = 13824
            eps: float = 1e-6
            patch_size: Tuple[int, int, int] = (1, 2, 2)
            num_heads: int = 40
            num_layers: int = 1
            has_image_input: bool = True
            has_image_pos_emb: bool = False

        with __patch("teletron.utils.set_config") as __mock_set_config, __patch(
            "teletron.utils.get_args"
        ) as __mock_get_args:
            from teletron.models.teleai import ParallelTeleaiModel, TeleaiModel

            __args = __Mock()
            __args.recompute_method = "block"
            __args.recompute_granularity = "full"
            __args.recompute_num_layers = 1
            __args.activation_offload = True
            __args.num_layers = 1
            __args.num_attention_heads = 40
            __args.distributed_vae = False
            __args.consumer_models_num = 1
            __mock_get_args.return_value = __args

            __model_config = dict(
                dit=dict(
                    type="ParallelTeleaiModel",
                    config=dict(
                        has_image_input=True,
                        patch_size=[1, 2, 2],
                        in_dim=36,
                        dim=5120,
                        ffn_dim=13824,
                        freq_dim=256,
                        text_dim=4096,
                        out_dim=16,
                        num_heads=40,
                        num_layers=1,
                        eps=1e-6,
                        has_image_pos_emb=False,
                    ),
                )
            )
            __mock_set_config.return_value = {"model_config": __model_config}

            initialize_model_parallel_base(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
                distributed_timeout_minutes=30,
            )

            teleaiConfig = _TeleaiParams()
            torch.manual_seed(1234)
            model = TeleaiModel(**asdict(teleaiConfig)).cuda(cuda_rank).to(torch.bfloat16)

            __cfg = __Mock(spec=__TransformerConfig)
            __cfg._cpu_offloading_context = None
            __cfg.perform_initialization = True
            __cfg.use_cpu_initialization = True
            __cfg.params_dtype = torch.bfloat16
            __cfg.gradient_accumulation_fusion = False
            __cfg.expert_model_parallel_size = 1
            __cfg.defer_embedding_wgrad_compute = False
            __cfg.async_tensor_model_parallel_allreduce = False
            __cfg.num_layers = __args.num_layers
            __cfg.sequence_parallel = False

            torch.manual_seed(1234)
            parallel_teleai_model = ParallelTeleaiModel(__cfg).cuda(cuda_rank).to(torch.bfloat16)
            def __tp_load_state_dict(base_model):
                base_dict = base_model.state_dict()
                tp_dict = {}
                col_w = ["self_attn.query.weight", "self_attn.key.weight", "self_attn.value.weight","ffn.0.weight",
                         "cross_attn.query.weight", "cross_attn.key.weight", "cross_attn.value.weight",
                         "cross_attn.img_key.weight", "cross_attn.img_value.weight"]
                col_b = ["self_attn.query.bias", "self_attn.key.bias", "self_attn.value.bias","ffn.0.bias",
                         "cross_attn.query.bias", "cross_attn.key.bias", "cross_attn.value.bias",
                         "cross_attn.img_key.bias", "cross_attn.img_value.bias"]
                row_w = ["ffn.2.weight", "self_attn.out_proj.weight",
                         "cross_attn.out_proj.weight"]
                norm_w = ["self_attn.norm_query.weight", "self_attn.norm_key.weight",
                          "cross_attn.norm_query.weight", "cross_attn.norm_key.weight",
                          "cross_attn.norm_image_key.weight"]
                def tp_col_weight_load(tp_dict, name, param):
                    r = mpu.get_tensor_model_parallel_rank()
                    s = mpu.get_tensor_model_parallel_world_size()
                    size = param.shape[0] // s
                    tp_dict[name] = param[r*size:(r+1)*size,:]
                def tp_col_bias_load(tp_dict, name, param):
                    r = mpu.get_tensor_model_parallel_rank()
                    s = mpu.get_tensor_model_parallel_world_size()
                    size = param.shape[0] // s
                    tp_dict[name] = param[r*size:(r+1)*size]
                def tp_row_weight_load(tp_dict, name, param):
                    r = mpu.get_tensor_model_parallel_rank()
                    s = mpu.get_tensor_model_parallel_world_size()
                    size = param.shape[1] // s
                    tp_dict[name] = param[:, r*size:(r+1)*size]
                def tp_norm_weight_load(tp_dict, name, param):
                    r = mpu.get_tensor_model_parallel_rank()
                    s = mpu.get_tensor_model_parallel_world_size()
                    size = param.shape[0] // s
                    tp_dict[name] = param[r*size:(r+1)*size]
                for name, param in base_dict.items():
                    if any(cw in name for cw in col_w):
                        tp_col_weight_load(tp_dict, name, param)
                    elif any(cb in name for cb in col_b):
                        tp_col_bias_load(tp_dict, name, param)
                    elif any(rw in name for rw in row_w):
                        tp_row_weight_load(tp_dict, name, param)
                    elif any(nw in name for nw in norm_w):
                        tp_norm_weight_load(tp_dict, name, param)
                    else:
                        tp_dict[name] = param
                return tp_dict
            parallel_teleai_model.load_state_dict(__tp_load_state_dict(model))
            input_dict = torch.load("/nvfile-heatstorage/ai_infra/data/lit117/teletron-testing/test_data/saved_inputs_360/input_dict_iter0_rank0.pt", map_location=f"cuda:{cuda_rank}")
            _ = model(x=input_dict['noisy_latents'],
                      timestep=input_dict['timestep'],
                      context=input_dict['prompt_emb']['context'],
                      clip_feature = input_dict['image_emb']['clip_feature'],
                      y=input_dict['image_emb']['y'])
            _ = parallel_teleai_model(x=input_dict['noisy_latents'],
                      timestep=input_dict['timestep'],
                      context=input_dict['prompt_emb']['context'],
                      clip_feature = input_dict['image_emb']['clip_feature'],
                      y=input_dict['image_emb']['y'])

            base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            param_names = {param: name for name, param in model.named_parameters()}
            timers = NoopTimer()
            ds_optim = DeepSpeedZeroOptimizer(
                base_optimizer,
                param_names,
                timers=timers,
                static_loss_scale=1.0,
                dynamic_loss_scale=False,
                dynamic_loss_args=None,
                clip_grad=0.0,
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
                gradient_accumulation_steps=1,
                ignore_unused_parameters=True,
                partition_grads=True,
                round_robin_gradients=False,
                has_moe_layers=False,
                fp16_master_weights_and_gradients=False,
                elastic_checkpoint=False,
            )

        # Prepare checkpoint files in DS dir: latest, model_optim_rng.pt, zero2 shards.
        model_ckpt_path = get_checkpoint_name(load_dir, iteration, release=False, return_base_dir=False)
        
        # Mock 64 shards for testing parallel load
        with __patch("megatron.core.mpu.get_data_parallel_world_size", return_value=64):
            optim_paths = get_checkpoint_name(load_dir, iteration, release=False, return_base_dir=False, use_zero2=True)

        from megatron.core import tensor_parallel as __tensor_parallel
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "rng_tracker_states": __tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        rng_state_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group(with_context_parallel=True),
        )

        if rank == 0:
            tracker = get_checkpoint_tracker_filename(load_dir)
            os.makedirs(load_dir, exist_ok=True)
            parent_dir = os.path.dirname(model_ckpt_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(tracker, "w") as f:
                f.write(str(iteration))
            torch.save(
                {
                    "iteration": iteration,
                    "args": types.SimpleNamespace(
                        consumed_train_samples=0,
                        consumed_valid_samples=0,
                        dit_world_size=world_size,
                    ),
                    "model": model.state_dict(),
                    "opt_param_scheduler": {"num_steps": 1},
                    "rng_state": rng_state_list,
                },
                model_ckpt_path,
            )
            for pth in optim_paths:
                os.makedirs(os.path.dirname(pth), exist_ok=True)
                torch.save(ds_optim.state_dict(), pth)
            q.put(f"deepspeed_model_and_zero2_shards_written rank{rank} count={len(optim_paths)}")
        try:
            torch.distributed.barrier(device_ids=[cuda_rank])
        except TypeError:
            torch.distributed.barrier()

        runtime_args = types.SimpleNamespace(
            load=load_dir,
            use_zero2=True,
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
            consumed_train_samples=0,
            consumed_valid_samples=0,
            auto_detect_ckpt_format=False,
            use_dist_ckpt=False,
            data_parallel_random_init=True,
            retro_add_retriever=False,
            lora=False,
            fp16=False,
            bf16=False,
            pretrained_checkpoint=None,
            exit_on_missing_checkpoint=False,
            use_distributed_optimizer=False,
            dit_world_size=world_size,
        )

        mixin = CheckPointMixin()

        with _patch("teletron.train.checkpoint.checkpoint.get_args", return_value=runtime_args), _patch(
            "teletron.train.checkpoint.checkpoint.update_num_microbatches", return_value=None
        ), _patch(
            "teletron.train.checkpoint.checkpoint.sys.exit",
            side_effect=RuntimeError("sys.exit called during load_checkpoint"),
        ), _patch("megatron.core.mpu.get_data_parallel_world_size", return_value=64):
            it, _flops, _opt, _sched = mixin.load_checkpoint([model], ds_optim, None)
        try:
            device_ok = False
            found = False
            for attr in ("fp32_partitioned_groups_flat", "fp16_groups_flat"):
                t = getattr(ds_optim, attr, None)
                if isinstance(t, list) and len(t) > 0 and torch.is_tensor(t[0]):
                    found = True
                    device_ok = t[0].is_cuda
                    break
            q.put(f"deepspeed_optimizer_cuda rank{rank} cuda_ok={device_ok}")
            if found and not device_ok:
                raise RuntimeError("deepspeed optimizer tensors not on cuda")
        except Exception as e:
            q.put(f"deepspeed_optimizer_cuda_check_exception rank{rank} error={str(e)}")
        q.put(f"deepspeed_load_checkpoint_ok rank{rank} iter={it}")
    except Exception as e:
        q.put(
            f"deepspeed_load_checkpoint_exception rank{rank} cuda={cuda_rank} "
            f"world_size={world_size} error={str(e)}"
        )
        import traceback

        traceback.print_exc()
    finally:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass


def _parallel_zero2_load(rank, world_size, q, tp_size, cp_size):
    from unittest.mock import Mock
    import datetime
    with patch("teletron.utils.get_args") as mock_get_args:
        args = Mock()
        args.distributed_vae = False
        args.consumer_models_num = 1
        args.distributed_vae_world_size = 0
        mock_get_args.return_value = args
    from teletron.core.parallel_state import initialize_model_parallel_base
    # Prefer a resilient device mapping: default to 0,1,2,3 and fall back to modulo mapping
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if not torch.cuda.is_available():
        q.put(f"cuda_unavailable rank{rank}")
        return
    device_count = torch.cuda.device_count()
    if device_count == 0:
        q.put(f"no_cuda_devices_visible rank{rank}")
        return
    cuda_devices = list(range(device_count))
    cuda_rank = cuda_devices[rank % device_count]
    torch.cuda.set_device(cuda_rank)
    q.put(f"worker_start_zero2 rank{rank} cuda={cuda_rank}")
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=3),
        )
        q.put(f"zero2_pg_ready rank{rank} world_size={world_size}")
        with patch("teletron.utils.get_args", return_value=args):
            initialize_model_parallel_base(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=cp_size,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
                distributed_timeout_minutes=30,
            )
        try:
            q.put(f"begin_sanity rank{rank}")
            sanity = torch.ones(1, device=f"cuda:{cuda_rank}") * (rank + 1)
            torch.distributed.all_reduce(sanity, op=torch.distributed.ReduceOp.SUM)
            expected = sum(range(1, world_size + 1))
            if int(sanity.item()) != expected:
                q.put(f"sanity_allreduce_fail rank{rank}")
                return
            q.put(f"sanity_allreduce_success rank{rank}")
        except Exception as e:
            q.put(f"sanity_allreduce_exception rank{rank} error={str(e)}")
            return

        base_dir = os.environ.get("TELETRON_OPTIM_DIR", "")
        cp_world = mpu.get_context_parallel_world_size()

        pattern = os.path.join(base_dir, "zero2_optim_*.pt")
        all_files = sorted(glob.glob(pattern))

        desired_dp_count = int(os.environ.get("DESIRED_DP_COUNT", "8"))
        target_count = desired_dp_count * cp_world
        q.put(f"zero2_files_found rank{rank} total_found={len(all_files)} target_count={target_count} cp_world={cp_world}")

        if rank == 0:
            print(f"[Init] Found {len(all_files)} weight files in {base_dir}. Target count: {target_count}")
            if len(all_files) > 0:
                print(f"[Init] Sample files: {[os.path.basename(f) for f in all_files[:3]]}")

        if len(all_files) >= target_count:
            file_paths = all_files[:target_count]
        else:
            file_paths = all_files
            if rank == 0:
                print(f"[Init] WARNING: Found fewer files ({len(file_paths)}) than expected ({target_count})")

        q.put(f"debug_rank{rank}_files_count={len(file_paths)}")

        mixin = CheckPointMixin()
        state_dict = {}
        t0 = time.time()

        q.put(f"debug_rank{rank}_start_load")
        try:
            mixin.load_zero2_optimizer(file_paths, state_dict)
            q.put(f"debug_rank{rank}_end_load_success")
        except Exception as e:
            q.put(f"debug_rank{rank}_load_failed error={str(e)}")
            raise

        t1 = time.time()

        total_files = len(file_paths)
        start_idx = (total_files * rank) // world_size
        end_idx = (total_files * (rank + 1)) // world_size
        q.put(f"zero2_shard_plan rank{rank} start={start_idx} end={end_idx} total={total_files} world_size={world_size}")
        q.put(
            f"zero2_rank_files rank{rank} files="
            f"{[os.path.basename(file_paths[i]) for i in range(start_idx, end_idx)]}"
        )

        loaded_indices = []
        failed_indices = []

        for i in range(start_idx, end_idx):
            if i < len(state_dict["optimizer"]) and state_dict["optimizer"][i] is not None:
                loaded_indices.append(i)
            else:
                failed_indices.append(i)

        unexpected_indices = []
        for i in range(total_files):
            if (i < start_idx or i >= end_idx) and i < len(state_dict["optimizer"]):
                if state_dict["optimizer"][i] is not None:
                    unexpected_indices.append(i)

        if len(failed_indices) == 0:
            loaded_names = [os.path.basename(file_paths[i]) for i in loaded_indices]
            size_bytes = sum(os.path.getsize(file_paths[i]) for i in loaded_indices)
            q.put(f"zero2_shards_loaded rank{rank} loaded_count={len(loaded_indices)} loaded_size={size_bytes}B")
            q.put(
                f"zero2_load_16_success rank{rank} count={len(loaded_indices)} files={loaded_names} "
                f"size={size_bytes}B duration={t1-t0:.3f}s"
            )
            if len(unexpected_indices) > 0:
                q.put(f"zero2_load_16_warning rank{rank} unexpected_loaded={unexpected_indices}")
        else:
            q.put(f"zero2_load_16_missing_files rank{rank} missing={failed_indices} unexpected={unexpected_indices}")
    except Exception as e:
        q.put(f"zero2_load_16_exception rank{rank} {str(e)}")
        return
    finally:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

def _spawn(world_size, fn, *args):
    def get_free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(get_free_port())
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    processes = []
    for i in range(world_size):
        p = ctx.Process(target=fn, args=(i, world_size, q) + args)
        p.start()
        processes.append(p)

    join_timeout_s = float(os.environ.get("SPAWN_JOIN_TIMEOUT_S", "600"))
    deadline = time.time() + max(join_timeout_s, 1.0)
    for p in processes:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        p.join(remaining)

    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
    for p in processes:
        try:
            p.join(5)
        except Exception:
            pass

    return q

def _drain_queue(q, label, total_timeout_s=60, idle_timeout_s=2):
    res = []
    start_t = time.time()
    last_msg_t = start_t
    while True:
        now = time.time()
        if now - start_t > total_timeout_s:
            break
        if now - last_msg_t > idle_timeout_s:
            break
        try:
            msg = q.get(timeout=0.5)
            res.append(msg)
            print(f"[{label}] {msg}", flush=True)
            last_msg_t = time.time()
        except pyqueue.Empty:
            continue
    return res


class TestZero2RealLoad(TestCase):
    def test_real_dir_load_16(self):
        os.environ["TELETRON_OPTIM_DIR"] = "/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00"
        world_size = 4
        tp = 2
        cp = 2
        q = _spawn(world_size, _parallel_zero2_load, tp, cp)
        res = _drain_queue(q, "MainProcess/zero2", total_timeout_s=120, idle_timeout_s=5)
        self.assertGreater(len(res), 0, "no worker messages received")
        sanity = [r for r in res if r.startswith("sanity_allreduce_success")]
        self.assertEqual(len(sanity), world_size)
        loads = [r for r in res if r.startswith("zero2_load_16_success")]
        self.assertEqual(len(loads), world_size)
        # multi4_ok = [r for r in res if r.startswith("multi_load4_success")]
        # self.assertEqual(len(multi4_ok), 16)

    def test_two_card_load_8(self):
        os.environ["TELETRON_OPTIM_DIR"] = "/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00"
        os.environ["DESIRED_DP_COUNT"] = "4"
        world_size = 2
        tp = 1
        cp = 2
        q = _spawn(world_size, _parallel_zero2_load, tp, cp)
        res = _drain_queue(q, "MainProcess/zero2-2cards", total_timeout_s=120, idle_timeout_s=5)
        self.assertGreater(len(res), 0, "no worker messages received")
        sanity = [r for r in res if r.startswith("sanity_allreduce_success")]
        self.assertEqual(len(sanity), world_size)
        load_ok = [r for r in res if r.startswith("zero2_load_16_success")]
        load_miss = [r for r in res if r.startswith("zero2_load_16_missing_files")]
        self.assertEqual(len(load_ok) + len(load_miss), world_size)
        # multi4_ok = [r for r in res if r.startswith("multi_load4_success")]
        # self.assertEqual(len(multi4_ok), 8)





class TestDeepSpeedZero2LoadCheckpoint(TestCase):
    def test_deepspeed_zero2_load_checkpoint_end_to_end(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            self.skipTest("need >=4 visible cuda devices")
        try:
            import deepspeed  # noqa: F401
        except Exception as e:
            self.skipTest(f"deepspeed unavailable: {e}")

        base_dir = tempfile.mkdtemp(prefix="ds_zero2_load_ckpt_")
        os.environ["DS_ZERO2_LOAD_CKPT_DIR"] = base_dir
        os.environ["DS_ZERO2_LOAD_CKPT_ITER"] = "1"
        
        world_size = 4
        q = _spawn(world_size, _parallel_deepspeed_zero2_load_checkpoint)
        res = _drain_queue(q, "MainProcess/ds_zero2_load_ckpt", total_timeout_s=600, idle_timeout_s=20)
        self.assertGreater(len(res), 0, "no worker messages received")
        unavailable = [r for r in res if r.startswith("deepspeed_unavailable")]
        if len(unavailable) > 0:
            self.skipTest(unavailable[0])
        ok = [r for r in res if r.startswith("deepspeed_load_checkpoint_ok")]
        self.assertEqual(len(ok), world_size)

# def _parallel_full_load(rank, world_size, q, tp_size, cp_size):
#     from unittest.mock import MagicMock, patch
#     import numpy as np
#     import random
#     import datetime

#     if "CUDA_VISIBLE_DEVICES" not in os.environ:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#     if not torch.cuda.is_available():
#         q.put(f"cuda_unavailable rank{rank}")
#         return
#     device_count = torch.cuda.device_count()
#     if device_count == 0:
#         q.put(f"no_cuda_devices_visible rank{rank}")
#         return

#     cuda_rank = 0 if world_size == 1 else (rank % device_count)
#     torch.cuda.set_device(cuda_rank)
#     q.put(f"worker_start_full_load rank{rank} cuda={cuda_rank}")

#     def _cleanup_dist():
#         try:
#             if torch.distributed.is_initialized():
#                 torch.distributed.destroy_process_group()
#         except Exception:
#             pass

#     try:
#         torch.distributed.init_process_group(
#             backend="nccl",
#             init_method="env://",
#             world_size=world_size,
#             rank=rank,
#             timeout=datetime.timedelta(minutes=3),
#         )
#         q.put(f"full_load_pg_ready rank{rank} world_size={world_size}")
#         sanity = torch.ones(1, device=f"cuda:{cuda_rank}") * (rank + 1)
#         torch.distributed.all_reduce(sanity, op=torch.distributed.ReduceOp.SUM)
#         expected = sum(range(1, world_size + 1))
#         if int(sanity.item()) != expected:
#             q.put(f"full_load_sanity_allreduce_fail rank{rank}")
#             _cleanup_dist()
#             return
#         q.put(f"full_load_sanity_allreduce_success rank{rank}")
#     except Exception as e:
#         q.put(f"full_load_dist_init_exception rank{rank} error={str(e)}")
#         _cleanup_dist()
#         return

#     # Prepare Mock Args
#     mock_args = MagicMock()
#     # Required args for load_checkpoint
#     mock_args.load = "dummy_load_path"
#     mock_args.use_zero2 = True
#     mock_args.no_load_optim = False
#     mock_args.no_load_rng = False
#     mock_args.finetune = False
#     mock_args.consumed_train_samples = 0
#     mock_args.consumed_valid_samples = 0
#     mock_args.auto_detect_ckpt_format = False
#     mock_args.use_dist_ckpt = False
#     mock_args.data_parallel_random_init = True
#     mock_args.retro_add_retriever = False
#     mock_args.lora = False
#     mock_args.fp16 = False
#     mock_args.bf16 = True
#     mock_args.distributed_vae = False
#     mock_args.consumer_models_num = 1
#     mock_args.distributed_vae_world_size = 0
#     mock_args.pretrained_checkpoint = None
#     mock_args.exit_on_missing_checkpoint = False
#     mock_args.use_distributed_optimizer = False
#     mock_args.no_save_optim = False
#     mock_args.no_save_rng = False
#     mock_args.dit_world_size = world_size # Assuming this is used for scheduler scaling

#     # Identify Real Files
#     base_dir = os.environ.get("TELETRON_OPTIM_DIR", "")
#     pattern = os.path.join(base_dir, "zero2_optim_*.pt")
#     all_optim_files = sorted(glob.glob(pattern))
    
#     # Filter files if DESIRED_DP_COUNT is set (optional, consistent with previous test)
#     desired_dp_count = int(os.environ.get("DESIRED_DP_COUNT", "8"))
#     target_count = desired_dp_count * max(int(cp_size), 1)
#     if len(all_optim_files) >= target_count:
#         all_optim_files = all_optim_files[:target_count]
#     q.put(
#         f"full_load_files rank{rank} found={len(all_optim_files)} desired_dp={desired_dp_count} "
#         f"cp_size={cp_size} target_count={target_count} world_size_sim={world_size}"
#     )
#     if rank == 0 and world_size > 1:
#         for r in range(world_size):
#             s = (len(all_optim_files) * r) // world_size
#             e = (len(all_optim_files) * (r + 1)) // world_size
#             q.put(f"full_load_rank_files rank{r} start={s} end={e} files={[os.path.basename(p) for p in all_optim_files[s:e]]}")
    
#     if rank == 0:
#         print(f"[FullLoad] Rank {rank} targeting {len(all_optim_files)} optimizer files from {base_dir}")
#         if len(all_optim_files) > 0:
#             print(f"[FullLoad] Sample optim files: {[os.path.basename(p) for p in all_optim_files[:3]]}")

#     # Prepare Mocks for Checkpoint Loading
#     model = [MagicMock()]
#     model[0].sharded_state_dict.return_value = {}
#     model[0].state_dict_for_save_checkpoint.return_value = {}
#     model_called = {"flag": False}
#     model[0].load_state_dict.side_effect = lambda *_args, **_kwargs: model_called.__setitem__("flag", True)

#     optimizer = _RealZero2Optimizer(rank=rank, q=q)
    
#     scheduler = MagicMock()
    
#     # Setup CheckPointMixin
#     mixin = CheckPointMixin()
    
#     # Prepare Patch Context
#     # We need to patch:
#     # 1. get_args -> mock_args
#     # 2. read_metadata -> (iteration, release)
#     # 3. get_checkpoint_name -> returns real paths for zero2 files, dummy for model
#     # 4. torch.load -> intercept model load, allow optimizer load
#     # 5. dist_checkpointing.check_is_distributed_checkpoint -> False
#     # 6. checkpoint_exists -> True (to bypass initial check)
    
#     original_torch_load = torch.load

#     with patch("teletron.train.checkpoint.checkpoint.get_args", return_value=mock_args), \
#          patch("teletron.train.checkpoint.checkpoint.read_metadata", return_value=(1000, False)), \
#          patch("teletron.train.checkpoint.checkpoint.dist_checkpointing.check_is_distributed_checkpoint", return_value=False), \
#          patch("teletron.train.checkpoint.checkpoint.checkpoint_exists", return_value=True), \
#          patch("teletron.train.checkpoint.checkpoint.update_num_microbatches", return_value=None), \
#          patch("teletron.train.checkpoint.checkpoint.sys.exit", side_effect=RuntimeError("sys.exit called during load_checkpoint")), \
#          patch("teletron.train.checkpoint.checkpoint.get_checkpoint_name") as mock_get_ckpt_name, \
#          patch("teletron.train.checkpoint.checkpoint.torch.load") as mock_torch_load, \
#          patch.object(CheckPointMixin, "_load_zero2_checkpoint") as mock_zero2_loader, \
#          patch("teletron.train.checkpoint.utils.get_args", return_value=mock_args), \
#          patch("megatron.core.mpu.get_data_parallel_rank", return_value=rank), \
#          patch("megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0), \
#          patch("megatron.core.mpu.get_pipeline_model_parallel_rank", return_value=0), \
#          patch("random.setstate") as mock_random_setstate, \
#          patch("numpy.random.set_state") as mock_np_set_state, \
#          patch("torch.set_rng_state") as mock_torch_set_rng_state, \
#          patch("torch.cuda.set_rng_state") as mock_cuda_set_rng_state, \
#          patch("teletron.train.checkpoint.checkpoint.tensor_parallel") as mock_tp:

#         # Mock get_checkpoint_name
#         def get_ckpt_name_side_effect(load_dir, iteration, release=False, return_base_dir=False, use_zero2=False, **kwargs):
#             if use_zero2:
#                 # Must return the list of real file paths
#                 return all_optim_files
#             if return_base_dir:
#                 return "dummy_base_dir"
#             return "dummy_model_optim_rng.pt"
#         mock_get_ckpt_name.side_effect = get_ckpt_name_side_effect

#         # Mock torch.load
#         def torch_load_side_effect(f, map_location=None, weights_only=False):
#             # Check if f is one of our real optimizer files
#             # Note: f could be a Path object or string
#             f_str = str(f)
#             if "zero2_optim_" in f_str and os.path.exists(f_str):
#                 return original_torch_load(f, map_location=map_location, weights_only=weights_only)
            
#             # Otherwise return dummy state dict for model/rng
#             # Construct a dummy state dict that load_checkpoint expects
#             dummy_state = {
#                 "iteration": 1000,
#                 "args": mock_args,
#                 "model": {}, # model load will use this
#                 "optimizer": [], # Will be ignored/overwritten by load_zero2_optimizer logic if use_zero2 is True? 
#                                  # No, load_zero2_optimizer modifies this dict in-place.
#                                  # But wait, load_checkpoint calls _load_zero2_checkpoint, which returns state_dict.
#                                  # _load_zero2_checkpoint loads model_checkpoint_name first.
#                                  # So this dictionary is what _load_zero2_checkpoint gets from torch.load(model_checkpoint_name)
#                 "rng_state": [ # rng_state list for data parallel ranks
#                      {
#                         'random_rng_state': random.getstate(),
#                         'np_rng_state': np.random.get_state(),
#                         'torch_rng_state': torch.get_rng_state(),
#                         'cuda_rng_state': torch.cuda.get_rng_state(),
#                         'rng_tracker_states': {'model-parallel-rng': [0]} 
#                      } for _ in range(world_size) # Assuming enough for DP
#                 ],
#                 "opt_param_scheduler": {"num_steps": 1},
#             }
#             return dummy_state
#         mock_torch_load.side_effect = torch_load_side_effect
#         # Patch _load_zero2_checkpoint to ensure Zero2 path always loads real optimizer shards
#         def fake_zero2_loader(load_dir, checkpoint_step=None):
#             state = torch_load_side_effect("dummy_model_optim_rng.pt")
#             total_files = len(all_optim_files)
#             start_idx = (total_files * rank) // world_size
#             end_idx = (total_files * (rank + 1)) // world_size
#             optimizer_state_list = [None] * total_files
#             shard_paths = [all_optim_files[i] for i in range(start_idx, end_idx)]
#             per_rank_procs = int(os.environ.get("ZERO2_SUBPROC_PER_RANK", "16"))
#             procs = min(max(per_rank_procs, 1), max(len(shard_paths), 1))
#             q.put(f"full_load_zero2_subproc rank{rank} procs={procs} tasks={len(shard_paths)}")
#             q.put(f"full_load_zero2_map_location rank{rank} mode=cpu_then_cuda device=cuda:{cuda_rank}")
#             ctx = mp.get_context("spawn")
#             with ctx.Pool(processes=procs) as pool:
#                 loaded_shards_cpu = pool.map(_load_zero2_file_cpu, shard_paths)
#             loaded_shards = [_move_to_cuda(sd, cuda_rank) for sd in loaded_shards_cpu]
#             for off, i in enumerate(range(start_idx, end_idx)):
#                 optimizer_state_list[i] = loaded_shards[off]
#             state["optimizer"] = optimizer_state_list
#             size_bytes = 0
#             for i in range(start_idx, end_idx):
#                 try:
#                     size_bytes += os.path.getsize(all_optim_files[i])
#                 except Exception:
#                     pass
#             q.put(
#                 f"full_load_zero2_shards rank{rank} start={start_idx} end={end_idx} total={total_files} "
#                 f"loaded_count={max(end_idx-start_idx,0)} loaded_size={size_bytes}B"
#             )
#             q.put(
#                 f"full_load_zero2_rank_files rank{rank} files="
#                 f"{[os.path.basename(all_optim_files[i]) for i in range(start_idx, end_idx)]}"
#             )
#             return state, "dummy_base_dir", False
#         mock_zero2_loader.side_effect = fake_zero2_loader
#         rng_states_called = {"flag": False}
#         mock_tp.get_cuda_rng_tracker.return_value.set_states.side_effect = lambda _s: rng_states_called.__setitem__("flag", True)
#         try:
#             iteration, num_flops, opt, sched = mixin.load_checkpoint(model, optimizer, scheduler)
#             q.put(f"full_load_return rank{rank} iter={iteration}")
#             if model_called["flag"]:
#                 q.put(f"model_state_dict_applied rank{rank}")
#             else:
#                 q.put(f"model_state_dict_missing rank{rank}")
#             if optimizer.called:
#                 q.put(f"optimizer_load_state_dict_called rank{rank}")
#             else:
#                 q.put(f"optimizer_load_state_dict_missing rank{rank}")
#             q.put(f"rng_basic_applied rank{rank} random={mock_random_setstate.called} numpy={mock_np_set_state.called} torch={mock_torch_set_rng_state.called} cuda={mock_cuda_set_rng_state.called}")
#             if rng_states_called["flag"]:
#                 q.put(f"rng_states_applied rank{rank}")
#             else:
#                 q.put(f"rng_states_missing rank{rank}")
#             q.put(f"full_load_ok rank{rank}")
#         except Exception as e:
#             q.put(f"full_load_exception rank{rank} error={str(e)}")
#             import traceback
#             traceback.print_exc()
#             _cleanup_dist()
#             return

#     gc.collect()
#     try:
#         torch.cuda.empty_cache()
#     except Exception:
#         pass
#     finally:
#         try:
#             if torch.distributed.is_initialized():
#                 torch.distributed.destroy_process_group()
#         except Exception:
#             pass

# class TestFullLoad(TestCase):
#     def test_full_load_checkpoint_flow(self):
#         os.environ["TELETRON_OPTIM_DIR"] = "/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00"
#         os.environ["DESIRED_DP_COUNT"] = "64"
#         os.environ["ZERO2_SUBPROC_PER_RANK"] = "16"
#         if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
#             self.skipTest("cuda unavailable")
#         world_size = 4
#         tp = 1
#         cp = 1
#         q = _spawn(world_size, _parallel_full_load, tp, cp)
#         res = _drain_queue(q, "MainProcess/full_load", total_timeout_s=180, idle_timeout_s=8)
#         self.assertGreater(len(res), 0, "no worker messages received")
#         ok = [r for r in res if r.startswith("full_load_ok")]
#         self.assertEqual(len(ok), world_size)
#         opt_called = [r for r in res if r.startswith("optimizer_load_state_dict_called")]
#         self.assertEqual(len(opt_called), world_size)
#         model_called = [r for r in res if r.startswith("model_state_dict_applied")]
#         self.assertEqual(len(model_called), world_size)
#         shard_msgs = [r for r in res if r.startswith("full_load_zero2_shards")]
#         self.assertEqual(len(shard_msgs), world_size)
#         per_rank_files = [r for r in res if r.startswith("full_load_zero2_rank_files")]
#         self.assertEqual(len(per_rank_files), world_size)
#         subproc_msgs = [r for r in res if r.startswith("full_load_zero2_subproc")]
#         self.assertEqual(len(subproc_msgs), world_size)
#         map_loc_msgs = [r for r in res if r.startswith("full_load_zero2_map_location")]
#         self.assertEqual(len(map_loc_msgs), world_size)
#         self.assertTrue(all("device=cuda:" in m for m in map_loc_msgs))
#         opt_dev_msgs = [r for r in res if r.startswith("optimizer_real_first_tensor_device")]
#         self.assertEqual(len(opt_dev_msgs), world_size)
#         self.assertTrue(all("device=cuda:" in m for m in opt_dev_msgs))
#         shards = []
#         totals = set()
#         expected_total = int(os.environ.get("DESIRED_DP_COUNT", "8")) * cp
#         for msg in shard_msgs:
#             parts = msg.split()
#             rank_part = parts[1]
#             rank = int(rank_part.replace("rank", ""))
#             kv = {}
#             for p in parts[2:]:
#                 if "=" in p:
#                     k, v = p.split("=", 1)
#                     kv[k] = v
#             start = int(kv["start"])
#             end = int(kv["end"])
#             total = int(kv["total"])
#             loaded_count = int(kv["loaded_count"])
#             totals.add(total)
#             shards.append((rank, start, end, loaded_count))
#         self.assertEqual(totals, {expected_total})
#         shards_sorted = sorted(shards, key=lambda x: x[1])
#         self.assertEqual(shards_sorted[0][1], 0)
#         for i, (_rank, start, end, loaded_count) in enumerate(shards_sorted):
#             self.assertEqual(end - start, loaded_count)
#             if i + 1 < len(shards_sorted):
#                 self.assertEqual(end, shards_sorted[i + 1][1])
#         self.assertEqual(shards_sorted[-1][2], expected_total)
