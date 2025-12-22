from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Type
from teletron.core.distributed.base_encoder import BaseEncoder
from teletron.models.teleai.teleai_encoder import TeleaiEncoder
from teletron.models.wan.encoder.wan_encoder import WanVideoEncoder
from teletron.models.longcat_video.encoder.longcat_video_encoder import LongcatVideoEncoder


_ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {}
_ENCODER_REGISTRY["teleai_encoder"] = TeleaiEncoder
_ENCODER_REGISTRY["wan_encoder"] = WanVideoEncoder
_ENCODER_REGISTRY["longcat_encoder"] = LongcatVideoEncoder

def register_encoder(name: str):
    """
    一个装饰器，用于将编码器类注册到全局注册表中。

    Args:
        name (str): 编码器的唯一名称。
    """
    def decorator(cls: Type[BaseEncoder]):
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"错误: 编码器 '{name}' 已被注册。")
        if not issubclass(cls, BaseEncoder):
            raise TypeError(f"错误: 注册的类 '{cls.__name__}' 必须是 BaseEncoder 的子类。")
        
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator

def get_encoder(name: str, device: torch.device, **kwargs: Any) -> BaseEncoder:
    """
    根据名称从注册表中获取并实例化一个编码器。

    Args:
        name (str): 要获取的编码器的名称。
        device (torch.device): 编码器将被初始化的设备。
        **kwargs: 传递给编码器构造函数的其他参数。

    Returns:
        BaseEncoder: 所请求编码器的实例化对象。
    
    Raises:
        ValueError: 如果请求的名称在注册表中不存在。
    """
    if name not in _ENCODER_REGISTRY:
        raise ValueError(f"错误: 编码器 '{name}' 未在注册表中找到。可用选项: {list(_ENCODER_REGISTRY.keys())}")
    
    encoder_class = _ENCODER_REGISTRY[name]
    return encoder_class(device=device, **kwargs)
