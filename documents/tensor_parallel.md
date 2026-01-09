# TensorParallel

## 适配方法
1. **模型张量并行改造：**
使用TensorParallelMixin中的enable_tensor_parallel系列方法来将模型中的线性层改为列并行线性层或行并行线性层。
关于这些接口的使用方式详见(TensorParallelMixin)[TODO:TensorParallelMixin接口文档]

```python
class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def __init__(...):
        DiTBlock.__init__(...)
        ...
        # from TensorParallelMixin
        self.enable_ffn_tensor_parallel(self.ffn, config)
        self.enable_self_attn_tensor_parallel(self.self_attn, config)
        self.enable_cross_attn_tensor_parallel(self.cross_attn, config)
```

2. 检查其他层是否受TP影响
我们给Wan适配TP时发现Wan的qk norm是在整个hidden dim上取平均（而不是在head dim上取平均），
这意味着TP情况下，qknorm层收到的hidden dim是切分后的，必须要做一次reduce同步TP group内的rms。 
qk norm的反向也要做处理且更复杂，详见[TensorParallelMixin.TeleParallelRMSNorm](#TeleParallelRMSNorm)

## 接口文档
(TODO)
### TeleParallelRMSNorm

