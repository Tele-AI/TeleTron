# Teletron 配置文件详细说明

本文档详细说明了Teletron训练框架中配置文件的结构和各参数含义，以`examples/teleai/config/teleai_i2v.py`为例进行说明。

## 配置文件整体结构

配置文件采用Python字典格式，主要包含以下几个部分：
- **全局参数**: 视频输出的基础设置
- **dataset**: 数据集配置
- **eval**: 评估数据配置  
- **sampler**: 数据采样器配置
- **model_config**: 模型配置（包含DiT模型和编码器）

## 1. 全局参数

```python
dst_size = (832, 480)    # 目标视频分辨率 (宽, 高)
dst_fps = 16             # 目标视频帧率
dst_num_frames = 81      # 目标视频帧数
```

**参数说明：**
- `dst_size`: 训练和推理时的目标视频分辨率，格式为(宽, 高)
- `dst_fps`: 目标帧率，影响视频的流畅度
- `dst_num_frames`: 每个视频样本的帧数，影响序列长度和内存使用

## 2. 数据集配置 (dataset)

### 2.1 基础配置

```python
dataset=dict(
    type="ClipDataset",              # 数据集类型
    serialize_data=False,            # 是否序列化数据
    data_path_list=[...],            # 数据文件路径列表
    filter_cfg=dict(...),           # 数据过滤配置
    transforms=[...],                # 数据变换配置
)
```

**参数说明：**
- `type`: 数据集类型，支持"ClipDataset"等
- `serialize_data`: 是否将数据序列化存储以提高加载速度
- `data_path_list`: JSON格式数据文件的路径列表，支持多个数据源

### 2.2 数据路径配置

```python
data_path_list=[
    "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
    "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_1.json",
    # 更多数据文件...
]
```

**配置说明：**
- 支持多个数据文件，框架会自动合并
- 可以通过注释掉文件路径启用或禁用数据
- 建议按数据集和时间进行分类组织

### 2.3 数据过滤配置 (filter_cfg)

训练启动时会根据数据元信息和filter_cfg的要求过滤去除不符合要求的数据，
具体的filter实现在teletron/datasets/clip_dataset.py filter_data方法。。
```python
filter_cfg=dict(
    dst_size=dst_size,                    # 目标分辨率
    dst_num_frames=dst_num_frames,        # 目标帧数
    dst_fps=dst_fps,                      # 目标帧率
    multiple=16,                          # 尺寸对齐倍数
    min_area=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),  # 最小面积
    optical_flow_th=1.5,                  # 光流阈值
    aesthetic_th=5,                       # 美学质量阈值
    bucket_size_th=4,                     # 桶大小阈值
    motion_th=0,                          # 运动阈值
    clearity_th=0.9,                      # 清晰度阈值
    laplacian_th=30,                      # 拉普拉斯清晰度阈值
    training_suitability_th=5.0,          # 训练适应性阈值
    area_th=dst_size[0] * dst_size[1] * (2 if dst_size[1] == 480 else 1),   # 面积阈值
)
```

**参数详解：**
- `multiple`: 确保尺寸是该数值的倍数，便于模型处理
- `min_area/area_th`: 过滤掉分辨率过小的视频
- `optical_flow_th`: 基于光流的运动检测阈值，过滤静态内容
- `aesthetic_th`: 美学质量评分阈值，过滤低质量内容
- `motion_th`: 运动强度阈值
- `clearity_th`: 清晰度阈值，过滤模糊内容
- `laplacian_th`: 拉普拉斯算子计算的清晰度阈值
- `training_suitability_th`: 训练适用性综合评分阈值


### 2.4 数据变换配置 (transforms)
选中一条视频数据会对数据做指定一系列变换再给到模型训练，这些变换用下面的transforms来指定。
具体的transforms实现在teletron/datasets/transform

```python
transforms=[
    dict(
        type="SampleImages",
        num_frames=dst_num_frames,
    ),
    dict(
        type="PromptGenerator", 
        clean_prompt=True,
        default_prompt_prob=0.1,
    ),
    dict(
        type="GenerateRawFirstRefImage",
    ),
    dict(
        type="PackInputs",
        deterministic=True,
        image_keys=["images"],
        embedding_keys=["raw_first_image"],
    ),
]
```

**变换类型说明：**

#### SampleImages
- **作用**: 从视频中采样指定数量的帧
- `num_frames`: 采样的帧数

#### PromptGenerator  
- **作用**: 生成或处理文本提示词
- `clean_prompt`: 是否清理提示词
- `default_prompt_prob`: 使用默认提示词的概率

#### GenerateRawFirstRefImage
- **作用**: 为I2V任务生成参考图像（第一帧）
- 用于Image-to-Video任务的条件输入

#### PackInputs
- **作用**: 裁切视频，并打包数据为模型输入格式
- `deterministic`: 是否使用确定性裁切，影响训练结果可复现性
- `image_keys`: 图像数据的键名列表
- `embedding_keys`: 嵌入数据的键名列表

## 3. 评估数据配置 (eval)

```python
eval=dict(
    data_path_list=[
        "/nvfile-heatstorage/cjf/share/export_to_clipdataset/istock/istock_0.json",
    ],
)
```

**说明：**
- 指定评估时使用的数据文件
- 通常使用较小的数据集以加速评估

## 4. 数据采样器配置 (sampler)

```python
sampler=dict(
    type="DefaultSampler",    # 采样器类型
    shuffle=False,            # 是否打乱数据
    seed=42,                  # 随机种子
    drop_last=True,           # 是否丢弃最后不完整的batch
    infinite=True,            # 是否无限循环数据
)
```

**参数说明：**
- `shuffle`: 控制数据是否随机打乱，影响训练收敛
- `seed`: 随机种子，确保实验可复现  
- `drop_last`: 是否丢弃不完整的batch，避免尺寸不匹配
- `infinite`: 无限循环数据集，适合长期训练

## 5. 模型配置 (model_config)

### 5.1 DiT模型配置

```python
dit=dict(
    type="ParallelTeleaiModel",    # 模型类型
    config=dict(
        has_image_input=True,      # 是否有图像输入 (I2V任务设为True)
        patch_size=[1, 2, 2],      # 补丁大小 [时间, 高, 宽]
        in_dim=36,                 # 输入维度 (I2V:36, T2V:16)
        dim=1536,                  # 隐藏维度 (1.3B:1536, 10B/14B:5120)
        ffn_dim=8960,              # FFN维度 (1.3B:8960, 10B/14B:13824)
        freq_dim=256,              # 频率编码维度
        text_dim=4096,             # 文本特征维度
        out_dim=16,                # 输出维度
        num_heads=12,              # 注意力头数 (1.3B:12, 10B/14B:40)
        num_layers=30,             # 层数 (1.3B:30, 10B:30, 14B:40)
        eps=1e-6,                  # 数值稳定性参数
        has_image_pos_emb=False,   # 是否使用图像位置编码
    ),
)
```

**模型规格对照表：**

| 参数 | 1.3B模型 | 5B模型  | 10B模型 | 14B模型 |
|------|---------|-------|-------|---------|
| in_dim | 36(I2V)/16(T2V) | 48  | 同1.3B | 同1.3B |
| dim | 1536 | 3072  | 5120  | 5120 |
| ffn_dim | 8960 | 14336 | 13824 | 13824 |
| num_heads | 12 | 24    | 40    | 40 |
| num_layers | 30 | 30    | 30    | 40 |

**关键参数说明：**
- `has_image_input`: I2V任务设为True，T2V任务设为False
- `patch_size`: [时间维度, 空间高度, 空间宽度] 的补丁分割大小
- `in_dim`: 根据任务类型设置，I2V包含图像信息所以维度更大

### 5.2 编码器配置

```python
encoder=dict(
    type="teleai_encoder",     # 编码器类型
    encoder_schema=['context', 'img_clip_feature', 'img_emb_y', 'latents'],  # 编码器需要计算的特征
    vae=dict(...),            # VAE配置
    text_encoder=dict(...),   # 文本编码器配置  
    image_encoder=dict(...),  # 图像编码器配置
    depth_model=dict(...),    # 深度模型配置
)
```

#### encoder_schema 说明
定义了编码器需要计算哪些特征，这些特征会给到DiT用于训练。不同任务DiT需要的输入特征不同：
- `context`: text prompt编码为上下文特征
- `img_clip_feature`: 图像prompt编码为CLIP特征  
- `img_emb_y`: 将图像编码为VAE潜在特征
- `latents`: 将视频编码为VAE潜在特征

#### VAE配置
```python
vae=dict(
    path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth",
    tiler_kwargs=dict(
        tiled=False,           # 是否使用分块编码
        tile_size=(34, 34),    # 分块大小
        tile_stride=(18, 16),  # 分块步长
    ),
    torch_compile = False
)
```

**参数说明：**
- `tiled`: 大分辨率视频可启用分块处理节省内存，81帧的情况下，一般1080P及以下分辨率不用tile效率比较高。
- `tile_size/tile_stride`: 控制分块的大小和重叠程度
- `torch_compile`（推荐开启）: 选择是否使用 torch.compile，加速 VAE 计算效率，以下是i2v场景各分辨率下开启compile的加速效果：

| shape | latents | img_emb_y | clip_feature | context | 其他 | 总和 | 提升比例 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (81, 1920, 1088) | 5372 | 4859 | 59 | 486 | 617 | 11393 | -19.61% |
| (81, 1280, 720) | 2099 | 1930 | 26 | 176 | 7 | 4238 | -37.17% |
| (81, 448, 784) | 794 | 720 | 13 | 33 | 5 | 1565 | -39.27% |
| (81, 368, 656) | 559 | 507 | 12 | 32 | 2 | 1112 | -37.25% |

#### 文本编码器配置
```python
text_encoder=dict(
    path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/google/umt5-xxl",
)
```

#### 图像编码器配置
```python
image_encoder=dict(
    path="/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    torch_compile = False
)
```
- `torch_compile`: 选择是否使用 torch.compile，加速 CLIP 计算效率

#### 深度模型配置
```python
depth_model=dict(
    path="/nvfile-heatstorage/ai_infra/ckpts/lit117/qiuyang/video_depth_anything_vitl.pth",
)
```

## 6. 配置文件自定义指南

### 6.1 切换任务类型

**T2V (Text-to-Video) 配置：**
```python
# 全局参数保持不变
config = dict(
    # 数据变换中移除GenerateRawFirstRefImage
    transforms=[
        dict(type="SampleImages", num_frames=dst_num_frames),
        dict(type="PromptGenerator", clean_prompt=True, default_prompt_prob=0.1),
        # 移除 GenerateRawFirstRefImage
        dict(type="PackInputs", deterministic=True, image_keys=["images"]),
    ],
    # 模型配置
    model_config=dict(
        dit=dict(
            config=dict(
                has_image_input=False,  # 改为False
                in_dim=16,              # 改为16
                # 其他参数保持不变
            )
        ),
        encoder=dict(
            encoder_schema=['context', 'latents'],  # 移除图像相关schema
            # 移除image_encoder和depth_model配置
        )
    )
)
```

### 6.2 调整模型规模

```python
# 切换到5B模型
dit=dict(
    config=dict(
        in_dim=48,      # 5B模型输入维度
        dim=3072,       # 隐藏维度
        ffn_dim=14336,  # FFN维度  
        num_heads=24,   # 注意力头数
        num_layers=30,  # 层数
    )
)
```

### 6.3 数据过滤调整

```python
# 更严格的质量过滤
filter_cfg=dict(
    optical_flow_th=2.0,        # 提高运动要求
    aesthetic_th=6.0,           # 提高美学要求
    clearity_th=0.95,           # 提高清晰度要求
    laplacian_th=40,            # 提高清晰度阈值
)

# 更宽松的质量过滤 (用于数据稀缺情况)
filter_cfg=dict(
    optical_flow_th=1.0,
    aesthetic_th=4.0, 
    clearity_th=0.8,
    laplacian_th=20,
)
```

## 7. 常见问题和注意事项

### 7.1 内存不足问题
- 减少`dst_num_frames`
- 降低`dst_size`分辨率
- 启用VAE的`tiled=True`

### 7.2 配置文件路径
配置文件需要放在`examples/{model}/config/`目录下，并在shell脚本中使用`config.{filename}.config`格式引用。
