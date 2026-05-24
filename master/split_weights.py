import torch
import os
import json
import gc
from safetensors.torch import load_file
from tqdm import tqdm

def split_sharded_qwen_moe(model_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "experts"), exist_ok=True)

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        print("未找到分片索引文件，请检查 model_path")
        return

    # 1. 读取索引，获取所有分片文件名
    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    # 找出所有唯一的分片文件名
    shard_files = sorted(set(index_data["weight_map"].values()))
    
    # 2. 准备容器（仅存放在内存中，最后保存）
    base_model_weights = {}
    experts_weights = {i: {} for i in range(60)} # Qwen1.5-MoE 有 60 个专家

    print(f"检测到 {len(shard_files)} 个权重分片，开始逐一处理...")

    # 3. 逐个分片处理，节省内存
    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        print(f"\n正在处理分片: {shard_name}")
        
        # 加载单个分片
        shard_sd = load_file(shard_path)
        
        for key in list(shard_sd.keys()):
            value = shard_sd[key]
            if ".mlp.experts." in key:
                # 提取层号和专家索引
                # key: model.layers.0.mlp.experts.15.down_proj.weight
                parts = key.split(".")
                expert_idx = int(parts[5])
                experts_weights[expert_idx][key] = value
            else:
                base_model_weights[key] = value
            
            # 从原始字典中删除引用，辅助垃圾回收
            del shard_sd[key]
        
        del shard_sd
        gc.collect()

    # 4. 物理保存
    print("\n--- 正在保存切分后的权重 ---")
    torch.save(base_model_weights, os.path.join(save_path, "base_model.pt"))
    print("✓ base_model.pt 已保存")
    del base_model_weights
    gc.collect()

    for i in tqdm(range(60), desc="保存专家文件"):
        if experts_weights[i]: # 确保该专家有参数
            torch.save(experts_weights[i], os.path.join(save_path, "experts", f"expert_{i}.pt"))
    
    print("\n切分完成！所有专家已保存至 experts/ 目录。")

if __name__ == "__main__":
    split_sharded_qwen_moe("/home/oai/moe-dis/weights/qwen1.5-moe-2.7b-chat", "/home/oai/moe-dis/split_weights_chat")
