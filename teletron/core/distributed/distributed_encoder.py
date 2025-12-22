import os
import torch
import torch.distributed as dist
import collections
import time
from typing import Callable, Any, Dict, List
from datetime import datetime
import psutil
import traceback
import copy
import random
import json
import numpy as np
import logging 
from teletron.core.parallel_state import get_comm_pair, get_world_group, CommPair
from teletron.utils import get_args, get_timers
from teletron.train.checkpoint import ensure_directory_exists
from teletron.models.encoder_registry import get_encoder

# --- 常量定义 ---
NUM_ITEMS_PER_CONSUMER = 100000
MAX_QUEUE_PER_CONSUMER_ON_PRODUCER = 2

TRAIN_MODE = 'train'
VALID_MODE = 'valid'

def cleanup_dist():
    """如果分布式环境已初始化，则销毁进程组 """
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: 销毁进程组 ")
        dist.destroy_process_group()

def merge_commpairs(commpairs: list) -> Dict[int, CommPair]:
    """
    将通信对列表（commpairs）根据相同的生产者和数据并行设置进行合并
    这用于将需要相同数据的消费者分组
    """
    merge_dict = {}
    for cp in commpairs:
        key = (cp.producer, cp.dp_rank, cp.dp_size)
        if key not in merge_dict:
            merge_dict[key] = []
        
        consumers = cp.consumer if isinstance(cp.consumer, list) else [cp.consumer]
        merge_dict[key].extend(consumers)
    
    merged_list = {}
    for idx, (key, consumers_list) in enumerate(merge_dict.items()):
        new_cp = CommPair(
            producer=key[0],
            consumer=sorted(list(set(consumers_list))),
            dp_rank=key[1],
            dp_size=key[2]
        )
        merged_list[idx] = new_cp
    return merged_list

def _set_random_seed_by_rank(seed_=1234):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different producer get different seeds.
        seed = seed_ + (10 * torch.distributed.get_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.device_count() > 0:
        #     tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))

class DistDataProducer:
    """
    分布式数据生产者
    该类负责从数据集中加载、编码数据，并通过 PyTorch Distributed 同步地发送给消费者进程
    """
    def __init__(
        self,
        rank: int,
        encoder_name: str,
        device,
        build_train_valid_test_data_iterators: Callable,
        train_ds: Any = None,
        valid_ds: Any = None,
    ):
        self.rank = rank
        self.device = device
        
        self._setup_logger() 
        self.logger.info("初始化开始...")

        self.args = get_args()
        _set_random_seed_by_rank(self.args.seed)
        self.encoder = get_encoder(name=encoder_name, device=self.device)
        self.build_data_iterators_fn = build_train_valid_test_data_iterators
        self.train_ds_preloaded = train_ds
        self.valid_ds_preloaded = valid_ds
        
        self.iteration = 0
        self.train_iteration = 0
        self.target_train_iters = self.args.train_iters
        self.batch_size = 1
        
        self.same_data_group = {}

        self.modes = [TRAIN_MODE]
        if self.args.eval_iters > 0:
            self.modes.append(VALID_MODE)
        self.in_train_epilogue = False
        self.logger.info(f"运行模式: {self.modes}")

        self.encoder.setup()
        self.logger.info("编码器设置完成")

        self.comm_pairs = get_comm_pair()
        self.merged_comm_pairs = merge_commpairs(self.comm_pairs)
        self.logger.info(f"原始通信对: {self.comm_pairs}")
        self.logger.debug(f"合并后通信对: {self.merged_comm_pairs}") # 使用 debug 级别，因为这个信息比较冗长
        
        self._initialize_consumer_state()
        self._create_data_iterators()
        self._initialize_queues_and_trackers()
        self._setup_profiler()

        # init timers
        self.timers = get_timers()
        self.timers.get_timer('encoder-once-time')

        self.logger.info("初始化完成")

    def _setup_logger(self):

        self.logger = logging.getLogger(f"ProducerRank{self.rank}")
        if not self.logger.handlers:

            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            # 定义 handler 的输出格式
            formatter = logging.Formatter(
                f'PRODUCER (Rank {self.rank}) [%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s',
                datefmt='%H:%M:%S'
            )
            ch.setFormatter(formatter)

            self.logger.addHandler(ch)

        self.logger.propagate = False

    def _get_gpu_memory_usage(self) -> str:
        """获取并格式化当前GPU显存使用情况"""
        try:
            if not torch.cuda.is_available():
                return "CUDA not available"
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            free, total = torch.cuda.mem_get_info(self.device)
            free_mb = free / 1024**2
            total_mb = total / 1024**2
            return (f"GPU Mem: Alloc={allocated:.2f}MB, "
                    f"Reserv={reserved:.2f}MB, Free={free_mb:.2f}MB, Total={total_mb:.2f}MB")
        except Exception as e:
            return f"GPU Mem: Error getting info - {e}"

    def _get_shm_usage(self) -> str:
        """获取并格式化/dev/shm使用情况"""
        try:
            shm_usage = psutil.disk_usage("/dev/shm")
            used_mb = shm_usage.used / 1024**2
            total_mb = shm_usage.total / 1024**2
            return f"SHM Mem: Used={used_mb:.2f}MB, Total={total_mb:.2f}MB ({shm_usage.percent}%)"
        except (FileNotFoundError, AttributeError):
            return "SHM Mem: /dev/shm not found or psutil error."

    def _initialize_consumer_state(self):
        """与 Consumers 同步初始状态，如训练迭代步数 """
        self.logger.info("正在从 Consumers 获取初始状态...")
        num_consumers = len(self.comm_pairs)
        consumers_data = torch.zeros((num_consumers, 3), dtype=torch.int64, device=self.device)
        
        reqs = []
        for i, cp in enumerate(self.comm_pairs):
            consumer_rank = cp.consumer
            self.logger.debug(f"PRE-RECV from Consumer Rank {consumer_rank} for initial state...")
            reqs.append(dist.irecv(tensor=consumers_data[i], src=consumer_rank))

        for i, req in enumerate(reqs):
            consumer_rank = self.comm_pairs[i].consumer
            self.logger.debug(f"Waiting for Consumer Rank {consumer_rank}'s initial state...")
            req.wait()
            self.logger.info(f"POST-RECV: 已收到来自 Consumer Rank {consumer_rank} 的初始状态")
        self.args.iteration = consumers_data[0][0].item()
        self.args.consumed_train_samples = consumers_data[0][1].item() // self.args.distributed_vae_world_size 
        self.args.consumed_valid_samples = consumers_data[0][2].item()
        self.logger.info(f"状态同步完成 Iteration: {self.args.iteration}, Consumed Train: {self.args.consumed_train_samples}, Consumed Valid: {self.args.consumed_valid_samples}")

    def _create_data_iterators(self):
        """根据合并后的通信对创建数据迭代器 """
        self.logger.info("正在创建数据迭代器...")
        self.data_iterators = {}
        
        train_ds_current = self.train_ds_preloaded
        valid_ds_current = self.valid_ds_preloaded

        train_iter, valid_iter, _, train_ds_current, valid_ds_current = self.build_data_iterators_fn(
            is_tp_first=True, dp_rank=0, dp_size=1,
            train_ds_prev=train_ds_current, valid_ds_prev=valid_ds_current, return_ds=True
        )

        self.data_iterators[TRAIN_MODE] = train_iter
        if VALID_MODE in self.modes:
            self.data_iterators[VALID_MODE] = valid_iter

        self.logger.info("数据迭代器创建完成")

    def _initialize_queues_and_trackers(self):
        self.same_data_group = {}
        for idx, mcp in self.merged_comm_pairs.items():
            first_consumer = mcp.consumer[0]
            self.same_data_group[first_consumer] = mcp.consumer
        """初始化数据队列和发送/生产计数器 """
        self.logger.info("正在初始化队列和计数器...")
        all_consumer_ranks = [cp.consumer for cp in self.comm_pairs]
        self.data_queues = {}
        self.produced_count = {}
        self.sended_count = {}

        for mode in self.modes:
            self.data_queues[mode] = {rank: collections.deque() for rank in all_consumer_ranks}
            self.produced_count[mode] = {rank: 0 for rank in all_consumer_ranks}
            self.sended_count[mode] = {rank: 0 for rank in all_consumer_ranks}
        self.logger.info("队列和计数器初始化完成")

    def _setup_profiler(self):
        """如果配置中启用，则设置PyTorch Profiler """
        self.profiler = None
        if self.args.producer_profile:
            prof_save_path = os.path.join(self.args.profile_path, f"producer/rank_{self.rank}.json")
            ensure_directory_exists(prof_save_path)
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
                on_trace_ready=lambda p: p.export_chrome_trace(prof_save_path),
                record_shapes=True
            )
            self.logger.info(f"Profiler已设置，结果将保存到: {prof_save_path}")

    def _produce_and_enqueue_data(self, idx: int, mcp: CommPair, mode: str, raw_batch=None):
        """从数据迭代器生产数据，编码后放入队列 """
        first_consumer = mcp.consumer[0]
        
        if raw_batch is None:
            try:
                self.logger.debug(f"PRE-GET-RAW-DATA for mode [{mode}] iter [{idx}]")
                raw_batch = next(self.data_iterators[mode])
            except StopIteration:
                self.logger.warning(f"警告: {mode} 模式的数据迭代器 {idx} 已耗尽")
                return
        
        self.logger.debug(f"POST-GET-RAW-DATA for mode [{mode}] iter [{idx}]")
        self.logger.debug(f"PRE-ENCODE: {self._get_gpu_memory_usage()}")

        self.timers.start_timer('encoder-once-time')
        tensors_to_send = self.encoder.encode(raw_batch)
        self.timers.stop_timer('encoder-once-time')
        encode_time = self.timers.get_elapsed_time('encoder-once-time')
        
        self.logger.debug(f"POST-ENCODE: {self._get_gpu_memory_usage()}")
        
        self.produced_count[mode][first_consumer] += 1
        item_index = self.produced_count[mode][first_consumer]
        
        self.logger.info(f"mode [{mode}] iter [{idx}]: produced {item_index} data, encoded {raw_batch['images'].shape} data cost {encode_time:.3f}s")

        for consumer_rank in self.same_data_group[first_consumer]:
            self.data_queues[mode][consumer_rank].append(tensors_to_send)
            self.logger.debug(f"QUEUE for Consumer {consumer_rank}: push {item_index} data")

    def _send_data_from_queue(self, cp: CommPair, mode: str):
        """从队列中取出数据，并使用同步方式发送 """
        consumer_rank = cp.consumer
        
        self.sended_count[mode][consumer_rank] += 1
        item_index = self.sended_count[mode][consumer_rank]
        
        tensors_to_send = self.data_queues[mode][consumer_rank].popleft()
        self.logger.debug(f"QUEUE for Consumer {consumer_rank}: get {item_index} data for sending")
        
        meta_info = {key: val.shape for key,val in tensors_to_send.items()}
        packed_tensor = self.encoder._pack_tensors([tensors_to_send[key] for key in tensors_to_send.keys()])

        resource_status = f"{self._get_gpu_memory_usage()} | {self._get_shm_usage()}"
        self.logger.debug(f"PRE-SEND-META to Consumer {consumer_rank} (item {item_index}): {meta_info}. Status: {resource_status}")
        dist.send_object_list([meta_info], dst=consumer_rank)
        self.logger.debug(f"POST-SEND-META to Consumer {consumer_rank} (item {item_index}): success")

        self.logger.debug(f"PRE-SEND-TENSOR to Consumer {consumer_rank} (item {item_index}): shape={packed_tensor.shape}, dtype={packed_tensor.dtype}")
        dist.send(tensor=packed_tensor, dst=consumer_rank)
        self.logger.debug(f"POST-SEND-TENSOR to Consumer {consumer_rank} (item {item_index}): success")

    def _get_mode_to_process(self, in_train_epilogue=False):
        """确定当前步骤要处理的模式（训练或验证）"""
        if self.in_train_epilogue:
            assert getattr(self, "train_or_valid_mode", None) is not None 
            return self.train_or_valid_mode
        if VALID_MODE in self.modes:
            train_data_count = self.args.eval_interval
            eval_data_count = self.args.eval_iters
            num_sended_in_cycle = self.iteration % (train_data_count + eval_data_count)
            mode_to_process = TRAIN_MODE if num_sended_in_cycle < train_data_count else VALID_MODE
        else:
            mode_to_process = TRAIN_MODE
        return mode_to_process
    
    def _main_loop_step(self):
        """执行一个主循环步骤：生产和发送数据 """
        #1. 确定当前要处理的模式
        mode_to_process = self._get_mode_to_process()

        self.logger.debug(f"Start produce data for mode: {mode_to_process}")

        # 2. 生产数据
        # 先取一个数据
        raw_batch = next(self.data_iterators[mode_to_process])
        raw_batch_shape = raw_batch['images'].shape
        
        num_micro_batchs = self.args.global_batch_size // self.args.data_parallel_size
        for i in range(num_micro_batchs):
            self.logger.debug(f"Start num_micro_batch {i} produce data for mode: {mode_to_process}")
            for idx, mcp in self.merged_comm_pairs.items():
                if i == 0 and idx == 0:
                    self._produce_and_enqueue_data(idx, mcp, mode_to_process, raw_batch)
                else:
                    self._produce_and_enqueue_data(idx, mcp, mode_to_process)
            self.logger.debug(f"End num_micro_batch {i} produce data ")
            
            self.logger.debug(f"Start num_micro_batch {i} send data for mode: {mode_to_process}")
            for cp in self.comm_pairs:
                self._send_data_from_queue(cp, mode_to_process)
            self.logger.debug(f"End num_micro_batch {i} send data")

        # 3. 迭代计数器加一，每个iteration对应一个global batch
        self.iteration += 1
        if mode_to_process == TRAIN_MODE:
            self.train_iteration += 1
            
    def train_epilogue(self):
        if VALID_MODE in self.modes:
            self.in_train_epilogue = True 
            self.train_or_valid_mode = VALID_MODE
            for _ in range(self.args.eval_iters):
                self._main_loop_step()
            
    def run(self):
        """运行主循环，直到满足停止条件 """
        try:
            self.logger.info("主循环开始")

            while self.train_iteration < self.target_train_iters:
                if self.profiler:
                    if self.iteration == self.args.profile_step_start:
                        self.logger.info("启动性能分析器...")
                        self.profiler.start()
                    if self.iteration == self.args.profile_step_end:
                        self.logger.info("停止性能分析器...")
                        self.profiler.stop()
                        self.logger.info(f"性能分析数据已保存")

                self._main_loop_step()
                
                time.sleep(0.001)

            self.logger.info("所有 Consumer 已达到目标数据量. 主循环结束.")
            self.train_epilogue()
            self.logger.info("等待所有进程到达屏障...")
            dist.barrier(group=get_world_group())
            self.logger.info("所有进程已同步")

        except Exception as e:
            # [MODIFICATION] 使用 logger.exception 记录异常，它会自动包含堆栈跟踪
            self.logger.exception("!!!--- 主循环中发生严重异常 ---!!!")
            dist.abort(group=get_world_group())
        finally:
            self.logger.info("开始清理...")
            # 记录可能的退出时异常信息
            exc_info = traceback.format_exc()
            if "NoneType: None" not in exc_info: # 过滤掉正常的退出
                self.logger.error(f"清理阶段的异常信息: {exc_info}")
            cleanup_dist()
            self.logger.info("程序退出")
