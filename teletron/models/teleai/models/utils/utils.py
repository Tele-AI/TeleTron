import os

import safetensors
import torch

from contextlib import contextmanager
import hashlib

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def wrap_call(func):
    def f(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                continue

    return f


def get_device():
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:{}".format(int(os.environ.get("Teleai_MODELS_DEFAULT_DEVICE", "0")))
        )
    else:
        device = torch.device("cpu")
    return device

def save_state_dict(state_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".safetensors"):
        safetensors.torch.save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    # breakpoint()
    return hashlib.md5(keys_str).hexdigest()