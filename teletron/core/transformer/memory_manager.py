import torch
from collections import defaultdict, deque
from teletron.utils import get_args

class MemoryManager:
    """
    管理锁页内存缓冲区和专用的CUDA数据传输流。
    这对于实现高效的异步数据传输至关重要。
    """
    def __init__(self):
        self.device = torch.cuda.current_device
        self.memory_pool = defaultdict(deque)
        args = get_args()
        self.num_layers = args.num_layers
        self._warmup()

    def get_event(self):

        return torch.cuda.Event()

    def return_event(self, event):
        """将用完的 Event 返回池中"""
        pass
        

    def _warmup(self):
        pass

    def get_buffer(self,  shape, dtype):
        """从池中获取一个足够大的内存。"""
        key = (shape, dtype)
        if self.memory_pool[key]:
            # 从池中弹出一个可用的缓冲区
            return self.memory_pool[key].popleft()
        else:
            return torch.empty(shape, dtype=dtype)
        
    def return_buffer(self, buffer):
        """
        将使用完毕的锁页内存缓冲区归还到池中。
        """
        key = (buffer.shape, buffer.dtype)
        self.memory_pool[key].append(buffer)


# 全局管理器实例
_memory_manager = None

def get_memory_manager():
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager