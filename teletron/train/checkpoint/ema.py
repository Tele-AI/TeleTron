import functools
import json
import operator
from collections import OrderedDict
from typing import Union
from megatron.core import mpu
import torch
import torch.distributed as dist


class EMAModel:
    """Exponential Moving Average of models weights."""

    def __init__(
        self,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        optimization_step: int = 0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.decay = decay
        self.min_decay = min_decay
        self.optimization_step = optimization_step
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.rank = rank
        self.world_size = world_size

        self._param_dict = OrderedDict()
        self._shape_dict = OrderedDict()
        self._slice_dict = OrderedDict()

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, state_dict):
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        for name, param in state_dict.items():
            s_param = self._param_dict[name]
            s_slice = self._slice_dict[name]
            param = param.view(-1)
            pad_size = (
                self.world_size - param.numel() % self.world_size
            ) % self.world_size
            if pad_size > 0:
                param = torch.nn.functional.pad(param, [0, pad_size])
            param = param[s_slice[0] : s_slice[1]]
            s_param[:] = (s_param.float() * decay + param.float() * (1 - decay)).to(
                s_param.dtype
            )

    def load_state_dict(self, state_dict, device=None, dtype=None):
        self._param_dict = OrderedDict()
        self._shape_dict = OrderedDict()
        self._slice_dict = OrderedDict()
        for name, param in state_dict.items():
            self._shape_dict[name] = param.shape
            param = param.view(-1)
            pad_size = (
                self.world_size - param.numel() % self.world_size
            ) % self.world_size
            if pad_size > 0:
                param = torch.nn.functional.pad(param, [0, pad_size])
            local_size = param.numel() // self.world_size
            begin = self.rank * local_size
            end = (self.rank + 1) * local_size
            param = param[begin:end].clone()
            if device is not None:
                param = param.to(device)
            if dtype is not None:
                param = param.to(dtype)
            self._param_dict[name] = param
            self._slice_dict[name] = (begin, end)

    def state_dict(self, device=None, dtype=None):
        param_dict = OrderedDict()
        for name, s_param in self._param_dict.items():
            param_shape = self._shape_dict[name]
            if self.world_size > 1:
                dim_size = list(s_param.size())
                dim_size[0] = dim_size[0] * self.world_size
                s_params = torch.empty(dim_size, dtype=s_param.dtype, device=torch.cuda.current_device())
                torch.distributed._all_gather_base(s_params, s_param.contiguous(), group=mpu.get_data_parallel_group(with_context_parallel=True))
                param_size = functools.reduce(operator.mul, param_shape)
                param = s_params[:param_size].reshape(param_shape)
            else:
                param = s_param.reshape(param_shape)
            if device is not None:
                param = param.to(device)
            else:
                param = param.cpu()
            if dtype is not None:
                param = param.to(dtype)
            param_dict[name] = param
        return param_dict

    @property
    def config(self):
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
        }

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            config = json.load(open(config, "r"))
        return cls(**config)

    def save_config(self, save_path):
        with open(save_path, "w", encoding="utf-8") as writer:
            json_string = json.dumps(self.config, indent=2, sort_keys=True) + "\n"
            writer.write(json_string)

    def load_config(self, config):
        if not isinstance(config, dict):
            config = json.load(open(config, "r"))
        for key, val in config.items():
            setattr(self, key, val)
