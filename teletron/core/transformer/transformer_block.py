
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
# from megatron.core.tensor_parallel import checkpoint

class TransformerGeneralMixin:
    def enable_activation_offload(self, blocks):
        self.enable_activation_checkpointing(blocks)
        blocks.forward = self.activation_offload_forward_transformer_blocks(blocks)

    def activation_offload_forward_transformer_blocks(self, blocks):
        origin_forward = blocks.forward
        def wrap_blocks_with_offload(*args):
            return self._activation_offload_forward(origin_forward, *args)
        return wrap_blocks_with_offload

    def _activation_offload_forward(self, forward_blocks, *args):
        with torch.autograd.graph.save_on_cpu():
            return forward_blocks(*args)

    def enable_activation_checkpointing(self, blocks):
        from teletron.utils import get_args
        args = get_args()
        self.activation_recompute_method = args.recompute_method
        self.recompute_granularity = args.recompute_granularity
        self.recompute_num_layers = args.recompute_num_layers
        blocks.forward = self.checkpointed_forward_transformer_blocks(blocks)

    # todo: kwargs are not updated
    def checkpointed_forward_transformer_blocks(self, blocks):
        def wrap_blocks(*args):
            if self.recompute_granularity == "full"  and self.training:
                output = self._checkpointed_forward(blocks, *args)
            else:
                for block in blocks:
                    output = block(*args)
                    args = self._update_args(output, args)
            return output
        return wrap_blocks

    def _get_block(self, blocks, layer_number: int):
        return blocks[layer_number]    

    def _update_args(self, output, args):
        if isinstance(output, tuple):
            return output + args[len(output):]
        else:
            return (output,) + args[1:]

    def _checkpointed_forward(self, blocks, *args):

        def create_custom_forward(start, end, blocks):
            def custom_forward(*args):
                for index in range(start, end):
                    block = self._get_block(blocks, index)
                    output = block(*args)
                    args = self._update_args(output, args)
                return output
            return custom_forward

        if self.activation_recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            assert self.recompute_num_layers <= len(blocks)
            while _layer_num < len(blocks):
                output = checkpoint(
                    create_custom_forward(_layer_num, _layer_num + self.recompute_num_layers, blocks),
                    *args,
                    use_reentrant=False,
                )
                args = self._update_args(output, args)
                _layer_num += self.recompute_num_layers

        elif self.activation_recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for _layer_num in range(len(blocks)):
                if _layer_num < self.recompute_num_layers:
                    output = checkpoint(
                        create_custom_forward(_layer_num, _layer_num + 1, blocks),
                        *args,
                        use_reentrant=False,
                    )
                else:
                    block = self._get_block(blocks, _layer_num)
                    output = block(*args)
                args = self._update_args(output, args)
        else:
            raise ValueError(f"Invalid activation recompute method {self.activation_recompute_method}.")

        return output 

    def set_input_tensor(self, x):
        return None
