from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Tuple, List
from teletron.utils import get_args

def get_dtype(dtype_str: str):
    if dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float16':
        return torch.float16
    elif dtype_str == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported options are 'float32', 'float16', 'bfloat16'.")

class BaseEncoder(ABC):

    def __init__(self, device: torch.device, **kwargs: Any):
        
        args = get_args()
        self.device = device
        self.moe = (args.consumer_models_num > 1)
        self.dtype = get_dtype(args.encoder_dtype)

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """
        init models
        """
        pass

    @abstractmethod
    def encode(self, raw_batch: Dict[str, Any]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """

        Args:
            raw_batch (Dict[str, Any])

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
            - Tensor List (List[torch.Tensor])
            - Sizes of Tensors (torch.Tensor)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_output_schema() -> List[str]:
        """
        返回一个有序列表，其中包含编码器输出的所有张量的名称。
        这个顺序必须与 encode() 方法返回的张量列表的顺序严格一致。
        这是一个静态方法，因此无需实例化即可调用。

        返回:
            List[str]: 一个包含张量名称的字符串列表，例如 ['context', 'clip,'image_feature', 'latents']。
        """
        pass

    @staticmethod
    def _pack_tensors(tensors_to_pack: List[torch.Tensor], dtype=torch.bfloat16) -> torch.Tensor:
        """
        将一个张量列表展平并拼接成一个单一的扁平化张量。
        """
        if not tensors_to_pack:
            return torch.tensor([])
        
        flattened_tensors = [t.flatten().to(dtype) for t in tensors_to_pack]
        return torch.cat(flattened_tensors, dim=0)

    @staticmethod
    def _get_tensors_size(tensor_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        获取张量列表的形状信息，并将其作为一个整数张量返回。
        """
        size_info = ()
        for item in tensor_list:
            size_info += item.size()
        return torch.tensor(size_info, dtype=torch.int32, device=device)
    