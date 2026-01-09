import torch

# 1. 加载权重文件
weights1 = torch.load('/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/open_9b/node_0/release/mp_rank_00/model_optim_rng.pt' , map_location=torch.device('cuda'))  # 替换为你的文件路径
weights2 = torch.load('/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_moe_4_832_optim/node_0/release/mp_rank_00/model_optim_rng.pt', map_location=torch.device('cuda'))
weights2 = torch.load('/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_moe_4_1280p_optim/node_0/release/mp_rank_00/model_optim_rng.pt' , map_location=torch.device('cuda'))  # 替换为你的文件路径

# 2. 初始化距离统计
total_std_dist = 0.0
num_layers = 0
# breakpoint()

# 3. 逐层计算欧氏距离
for (name1, param1), (name2, param2) in zip(weights1['model'].items(), weights2['model'].items()):
    # 检查层名和形状是否一致
    name2 = name2[12:]
    if name1 != name2 or param1.shape != param2.shape:
        raise ValueError(f"层不匹配: {name1} vs {name2} | {param1.shape} vs {param2.shape}")
    
    # 展平张量为向量 [n]
    vec1 = param1.flatten()
    vec2 = param2.flatten()
    
    # 计算标准差（正则化的核心）
    std_dev = torch.sqrt(0.5 * (torch.norm(vec1) + torch.norm(vec2))) + 1e-8  # 避免除零

    # 计算正则化欧氏距离
    diff = (vec1 - vec2) / std_dev
    std_dist = torch.norm(diff, p=2)  # L2范数
    
    total_std_dist += std_dist.item()
    num_layers += 1
    print(f"层: {name1}:norm1: {torch.norm(vec1)}:norm2: {torch.norm(vec2)}: 正则化距离: {std_dist:.6f}")

# 4. 计算平均距离
if num_layers > 0:
    avg_std_dist = total_std_dist / num_layers
    print(f"\n平均正则化欧氏距离: {avg_std_dist:.6f}")
else:
    print("未找到可比较的层")