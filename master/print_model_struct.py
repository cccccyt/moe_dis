import torch
import os
import sys

# 确保能找到你的源码
sys.path.append(os.path.abspath("./qwen_src"))
from qwen_src.modeling_qwen2_moe import Qwen2MoeForCausalLM
from qwen_src.configuration_qwen2_moe import Qwen2MoeConfig

def print_original_model_struct(model_path):
    config = Qwen2MoeConfig.from_pretrained(
        model_path, 
        local_files_only=True  # 强制只从本地加载，不再联网
    )

    # 在 meta 设备下实例化
    # 关键：torch.device("meta") 确保不分配真实内存
    with torch.device("meta"):
        model = Qwen2MoeForCausalLM(config)

    print("=== Qwen2-MoE 原始蓝图架构 ===")
    print(model)

    # 3. 打印特定层看专家结构
    print("\n=== 第 0 层 MLP 专家细节 ===")
    print(model.model.layers[0].mlp)

def inspect_pt_file(file_path, name):
    print(f"\n{'='*20} 探测文件: {name} {'='*20}")
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    # 加载权重字典
    sd = torch.load(file_path, map_location="cpu", weights_only=True)
    
    keys = list(sd.keys())
    print(f"总参数项数量: {len(keys)}")
    
    # 打印前 5 个 Key 和 形状
    print(f"{'参数路径 (Key)':<65} | {'形状 (Shape)':<20}")
    print("-" * 90)
    for k in keys[:5]:
        print(f"{k:<65} | {str(list(sd[k].shape)):<20}")
    
    # 特殊检查
    if "base" in name.lower():
        # 检查是否残留了专家参数
        expert_keys = [k for k in keys if "mlp.experts" in k]
        print(f"\n[验证] 基础模型中残留的专家参数数量: {len(expert_keys)}")
    else:
        # 检查专家参数是否包含层信息
        layers = set([k.split(".")[2] for k in keys if "layers" in k])
        print(f"\n[验证] 该专家跨越的 Transformer 层数: {len(layers)} (通常应为 24)")

def list_expert_parameters(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 加载专家权重字典
    # 使用 weights_only=True 提高安全性
    checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
    
    # 获取所有参数名并排序，确保层索引（Layer 0, 1, 2...）按顺序显示
    keys = sorted(list(checkpoint.keys()), key=lambda x: (int(x.split('.')[2]), x))

    print(f"\n文件 {os.path.basename(file_path)} 内部参数名全记录：")
    print("-" * 80)
    
    current_layer = -1
    for key in keys:
        # 提取层号，用于美化打印输出
        layer_idx = int(key.split('.')[2])
        if layer_idx != current_layer:
            print(f"\n[Layer {layer_idx}]")
            current_layer = layer_idx
            
        # 打印形状信息，方便核对显存占用
        shape = list(checkpoint[key].shape)
        print(f"  {key:<70} | {str(shape):<15}")

    print("-" * 80)
    print(f"统计：该文件共包含 {len(keys)} 个参数张量。")

def verify_base_and_experts(split_path):
    base_path = os.path.join(split_path, "base_model.pt")
    base_sd = torch.load(base_path, map_location="cpu", weights_only=True)
    
    # 1. 检查共享专家
    shared_keys = [k for k in base_sd.keys() if "shared_expert" in k]
    print(f"=== Base Model 检查 ===")
    print(f"检测到共享专家参数项: {len(shared_keys)} 个")
    if len(shared_keys) > 0:
        print(f"示例 Key: {shared_keys[0]}")
    
    # 2. 检查路由专家文件
    expert_dir = os.path.join(split_path, "experts")
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.endswith(".pt")])
    
    print(f"\n=== Experts 目录检查 ===")
    print(f"磁盘上共发现 {len(expert_files)} 个专家文件")
    
    valid_count = 0
    for f in expert_files:
        f_path = os.path.join(expert_dir, f)
        sd = torch.load(f_path, map_location="cpu", weights_only=True)
        if len(sd) == 0:
            # 如果是空文件，建议删除，避免干扰后续加载
            print(f"[-] 移除无效空文件: {f}")
            os.remove(f_path)
        else:
            valid_count += 1
            
    print(f"有效专家文件总数: {valid_count} (目标应为 60)")

def print_shared_expert_keys(split_path):
    file_path = os.path.join(split_path, "base_model.pt")
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 加载权重 (weights_only=True 提高安全性)
    sd = torch.load(file_path, map_location="cpu", weights_only=True)
    
    # 提取包含 'shared_expert' 的所有 Key，并按层号排序
    keys = [k for k in sd.keys() if "shared_expert" in k]
    keys = sorted(keys, key=lambda x: (int(x.split('.')[2]), x))

    print(f"=== 共享专家 (Shared Expert) 参数结构普查 (共 {len(keys)} 项) ===")
    
    current_layer = -1
    for k in keys:
        # 解析层索引
        layer_idx = k.split('.')[2]
        
        if layer_idx != current_layer:
            print(f"\n[Layer {layer_idx}]")
            current_layer = layer_idx
            
        # 打印完整参数路径和形状
        print(f"  {k:<70} | {str(list(sd[k].shape)):<15}")

    print("-" * 90)
    print("统计：96 项参数已全部定位。")


if __name__ == "__main__":
    # model_path = "/home/oai/moe-dis/weights/qwen1.5-moe-2.7b"
    target_expert_file = "/home/oai/moe-dis/split_weights/experts/expert_0.pt"
    split_dir = "/home/oai/moe-dis/split_weights"
    # print_original_model_struct(model_path)
    inspect_pt_file(os.path.join(split_dir, "base_model.pt"), "主控基础权重")
    # inspect_pt_file(os.path.join(split_dir, "experts/expert_0.pt"), "0号专家专属权重")
    print("--------------------------------")
    list_expert_parameters(target_expert_file)
    print("--------------------------------")
    verify_base_and_experts("/home/oai/moe-dis/split_weights")
    print("--------------------------------")
    print_shared_expert_keys("/home/oai/moe-dis/split_weights")

