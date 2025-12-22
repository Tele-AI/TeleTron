# TensorParallel
## 适配方法

1. 改造DiT前向以切分和聚合序列。
```python
class ParallelTeleaiModel(ContextParallelMixin, TensorParallelMixin, TransformerGeneralMixin, TeleaiModel):
    def forward(...):
        ...
        x = self.split_input(x, dim=1)
        freqs = self.split_input(freqs, dim=0)
        x = self.blocks(x, context_emb, t_mod, freqs)
        x = self.gather_output(x, dim=1)
        ...
        return x 
```
2. 改造DiTBlock Attention层
调用ContextParallelMixin的enable_context_parallel方法使得self attention并行计算。
（一般cross attention不需要额外操作，因为kv没有切分，kv分别和切分的q计算attention即可）
```python
class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def __init__(self, *args, **kwargs):
        # from ContextParallelMixin
        self.enable_context_parallel(self.self_attn.attn)
```

3. 改造DiT block的modulate和gate层，在ContextParallelMixin中有适配了CP的modulate和gate层。
这两个层计算梯度后需要对shift、scale和gate的梯度额外在cp group做一次reduce sum，因为reduce前他们
只与当前cp rank的部分token做了梯度计算。
```python
class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def forward(self, x, context, t_mod, freqs):
        ...
        modulated_x1 = self.modulate_with_cp_grad_reduce(normed_x1, shift_msa, scale_msa)
        ...
        gated_x1 = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)
        ...
        modulated_x2 = self.modulate_with_cp_grad_reduce(normed_x2, shift_mlp, scale_mlp)
        ...
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, ffn_output)

        return x
```

4. 在DiT上应用反向hook。由于input sequence做了切分，大部分层是与切分的sequence做计算，因此他们的权重梯度也是部分结果，
需要在cp group做reduce sum。而部分层（如patch_emb、head）是对完全序列做的计算，不需要cp grad reduce，所以必须要在这里做特殊处理（不能合并到DP reduce）。

另外，modulation和time 相关的权重梯度，已经在modulate和gate中做了处理，所以这里也不需要额外reduce。适配新模型时需要注意。
（TODO：补充使用tensorwatch工具观测梯度以指导实现grad reduce的方法和案例，联系李天催更）

```python
    def register_cp_grad_reduce_hook(self):

        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook

        for name, param in self.named_parameters():
            if name.startswith("patch_emb") or \
                name.startswith("time") or \
                    name.startswith("head") or \
                    "modulation" in name:
                continue

            param.register_hook(self.cp_grad_reduce)

```

## 接口文档
TODO