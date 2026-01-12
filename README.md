# Teletron 使用文档

Teletron 是一个专为训练长上下文多模态Transformer模型而设计的分布式训练框架，支持多种视频生成模型的高效训练。

## News
* 2025-09-18：可以使用torch compile加速encoder编码了，各分辨率下有20-40%不等的encoder时延下降。详见config_guide.md中的vae配置部分。


## QuickStart

### 环境设置

一般可以直接使用basemodel的最新镜像。主要依赖包括torch、flash_attn、teleai_data_tool、yunchang、
deepspeed等，见requirements.txt

### shell脚本和配置文件py设置方法

#### 1. Shell脚本配置

在examples文件夹中，Teletron 为每个模型提供了shell脚本来启动训练流程，如examples/teleai/run.sh. 
以下是主要的配置参数：

请根据实际硬件资源（如GPU数量、节点数等）配置以下参数：

| 参数名               | 说明                                   |
|-------------------|--------------------------------------|
| `CP`              | 序列并行组大小，CP组内的GPU看到的是同一份数据的不同分片       |
| `TP`              | 张量并行长度，TP组内的GPU对模型线性层权重做分片           |
| `N_GPU_FOR_TRAIN` | 用于模型训练的 GPU 总数                       |
| `N_GPU_FOR_DATA`  | 用于数据服务的 GPU 数量                       |
| `N_LAYERS`        | 模型层数，默认为 25；调试时可设为 1（需与加载权重匹配）       |
| `N_MOE`           | MoE 模块数量，目前支持 1/2/4；为 1 时使用普通非 MoE 模型 |

注意，在P2P分布式编码器实现下，要求 N_GPU_FOR_TRAIN/CP/TP % N_GPU_FOR_DATA = 0。
在数据服务的分布式编码器实现下不强制要求整除，但我们依然有推荐的[最优配置比例](#训练效率和最优配置一览)。

配置示例：
```commandline
# Parallel config 
CP=2
TP=1

# Multi-node config 
N_MOE=1
N_GPU_FOR_TRAIN=16
N_GPU_FOR_DATA=8

# Single-node config 
N_MOE=1
N_GPU_FOR_TRAIN=1
N_GPU_FOR_DATA=1
```

**启动训练：**

推荐先在开发环境上使用单节点配置（N_GPU_FOR_TRAIN=4，N_GPU_FOR_DATA=1）启动单节点训练
```bash
# Teleai模型训练
bash examples/teleai/run.sh

# 根据任务自定义使用训练脚本和配置文件，如i2v任务则使用pretrain_i2v.py训练脚本和teleai_i2v配置文件
bash examples/teleai/run.sh examples/teleai/pretrain_i2v.py config.teleai_i2v.config

# Wan2.1模型训练
bash examples/wan/run_wan.sh

# 自回归Wan2.1模型训练
bash examples/wan/run_causal.sh
```

**训练配置参数：**
```bash
EXPR_NAME=i2v_1.3B           # 实验名称，会从这个路径加载和保存权重
TRAIN_SCRIPT=${1:-"examples/teleai/pretrain_i2v.py"}  # 训练脚本，默认为examples/teleai/pretrain_i2v.py，可以从命令行传入
CONFIG_PATH=${2:-"config.teleai_i2v.config"}  # 配置文件路径，默认为config.teleai_i2v.config，可以从命令行传入
```
配置文件在examples各个模型文件夹下的config文件夹，如examples/teleai/config/teleai_i2v.py，
注意传入时要用config.{文件名}.config这样的格式。


#### 2. Python配置文件设置

配置文件采用Python字典格式，主要包含数据集配置、DiT模型配置和encoder模型配置，其整体结构如下，
配置文件的详细说明见[config_guide.md](config_guide.md)：

```python
# 基础参数
dst_size = (832, 480)    # 目标分辨率
dst_fps = 16             # 目标帧率
dst_num_frames = 81      # 目标帧数

config = dict(
    # 数据集配置
    dataset=dict(
        type="ClipDataset",
        data_path_list=["/path/to/dataset.json"],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames=dst_num_frames,
            dst_fps=dst_fps,
            # 更多过滤参数...
        ),
        transforms=[...]
    ),
    
    # 模型配置
    model_config=dict(
        dit=dict(
            type="ParallelTeleaiModel",
            config=dict(
                has_image_input=False,
                patch_size=[1, 2, 2],
                dim=3072,
                num_heads=24,
                num_layers=30,
                # 其他模型参数...
            )
        ),
        encoder=dict(
            type="teleai_encoder",
            # 编码器配置...
        )
    ),
    sampler=dict(
        type="DefaultSampler",    
        # dataloader sampler 配置...
    )
)
```

### 模型支持列表

| 模型                | 参数量  | 输入维度 | 隐藏维度 | 注意力头数 | 层数 |
|-------------------|------|----|----------|------------|----|
| Teleai-1.3B       | 1.3B | 36/16 | 1536 | 12 | 30 |
| Teleai-5B         | 5B   | 48 | 3072 | 24 | 30 |
| Teleai-10B        | 10B  | 36/16 | 5120 | 40 | 30 |
| Teleai-14B        | 14B  | 36/16 | 5120 | 40 | 40 |
| CausalWan2.1-1.3B | 1.3B | 16 | 1536 | 12 | 30 |
其中除了CausalWan仅支持T2V之外，其他模型都支持i2v、multimask、sr和i2v_depth（带depth输入的i2v预训练）任务。

### 训练效率和最优配置一览
以下是1.3B 81帧 I2V任务的最优配置推荐（启用了FA3）

| 分辨率          | Data:Train | CP | 单step训练时间(s) | 单节点吞吐(FPS) | MFU |
|--------------|------------|----|--------------|-----------|-----|
| (384, 640)   | 1:1        | 1  | 1.77         | 182.4      | --  |
| (480, 720)   | 1:1        | 1  | 2.58         | 125.7      | --  |
| (720, 1280)  | 1:2        | 1  | 13.5         | 32.0      | --  |
| (1080, 1920) | 1:2        | 1  | 44.2         | 9.77      | --  |


以下是14B 81帧 I2V任务的最优配置推荐（启用了FA3）

| 分辨率          | Data:Train | CP | 单step训练时间(s) | 单节点吞吐(FPS) | MFU |
|--------------|------------|----|--------------|-----------|-----|
| (384, 640)   | 1:4        | 1  | 7.46         | 69.5      | --  |
| (480, 720)   | 1:4        | 1  | 11.9         | 43.6      | --  |
| (720, 1280)  | 1:8        | 8  | 9.24         | 7.78      | --  |
| (1080, 1920) | 1:8        | 8  | 34.7         | 2.07      | --  |

## 常用特性

### 分布式多模态编码器

分布式多模态编码器是Teletron的核心组件，支持将视频、图像和文本编码任务在独立的GPU上与DiT并行执行。

#### 使用方法

在shell脚本中启用分布式编码器：

```bash
MODEL_PARALLEL_ARGS=(
    --distributed-vae
    --distributed-vae-world-size $N_GPU_FOR_DATA
)
```

编码器配置示例：
```python
encoder=dict(
    type="teleai_encoder",  # 或 "wan_encoder"
    encoder_schema=['context', 'latents'],
    vae=dict(
        type="TeleaiVideoVAE_2_1",
        path="/path/to/vae.pth",
        tiler_kwargs=dict(
            tiled=False, # 2K以下分辨率推荐tiled=False
            tile_size=(34, 34),
            tile_stride=(18, 16),
        ),
    ),
    text_encoder=dict(
        path="/path/to/text_encoder.pth",
        tokenizer_path="/path/to/tokenizer",
    )
)
```

### ContextParallel（上下文并行）

ContextParallel 是专门为长序列训练设计的并行策略，将长序列分割到不同GPU上并行处理。

#### 使用方法

启用上下文并行，注意CP-size要能被模型的attention num head整除。
```bash
--context-parallel-size 2  # 设置CP大小
```

```commandline
Note：现在CausalWan还没有实现CP
```


### zero2分布式优化器

基于ZeRO-2的分布式优化器，将优化器状态分片以节省内存。

#### 使用方法

启用分布式优化器：
```bash
--use-zero2
```

### EMA

EMA用于模型权重的平滑更新，提高最终模型质量。在每次保存模型权重时额外保存一份ema权重，断点续训时也会加载。
基本不影响训练速度，但是会额外占用一点显存。目前的实现是把ema权重在所有训练的GPU上切分，所以训练GPU越多显存影响越小。

使用方法：
```bash
--with-ema
--ema-decay 0.999 # 一般设置0.999到0.9999
```

### 断点续训

支持训练中断后的恢复。

使用方法：
```bash
--save-interval 500          # 每500步保存一次
--save $CHECKPOINT_PATH_SAVE # 保存路径
--load $CHECKPOINT_PATH_LOAD # 加载路径
```

支持使用--override-opt_param-scheduler来用当前指定的超参（如lr、wd）覆盖上一次训练的优化器超参，
也可以使用--no-load-optim和--no-load-rng来跳过加载优化器状态或者rng state。

推荐使用--data-parallel-random-init训练，因为这样可以让不同rank随机采样的timestep不同，有利于稳定训练。
（开启后模型checkpoint中会额外存每份dp的rng state）


### TensorParallel（张量并行）

张量并行将模型参数分布到多个GPU上，实现模型级别的并行训练。

#### 使用方法

启用张量并行，要求TP-size*CP-size要能被模型的attention num head整除。
```bash
--tensor-model-parallel-size 2  # 设置TP大小
```
```commandline
Note：因为CP用的是ulysses实现，是在head维度切分，而TP切分hidden-size维度，
体现到attention上也会导致head维度被切分，因此要求TPxCP能被head数整除。
Note：现在CausalWan和Hunyuan还没有实现TP
```
### torch.compile 加速
#### 使用方法
启动 torch.compile，在VAE 模型上有显著的加速效果
参照 example/teleai/config 中的设置，在vae配置中加入torch_compile=True
```python
vae=dict(
    ...
    torch_compile = True
)
```
