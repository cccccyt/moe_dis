import torch
import torch.nn as nn
import os
import sys
import ray
import gc
import time
import threading

# 确保能找到 Qwen 源码（见 README.md: worker 机器需将 qwen_src/ 放在 worker_node.py 同目录）
_worker_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_worker_dir, "qwen_src")):
    sys.path.insert(0, _worker_dir)
from qwen_src.modeling_qwen2_moe import Qwen2MoeMLP

@ray.remote(num_gpus=1, max_concurrency=4)  # 允许最多4个请求并发处理
class RemoteExpertNode:
    def __init__(self, config, split_weights_path, expert_start=0, expert_end=60):
        """
        智能内存管理版 - 动态分配GPU/CPU专家，预留25%显存用于推理计算
        支持并发处理(max_concurrency=4)
        - GPU专家：在GPU上并发计算（高速）
        - CPU专家：在CPU上并发计算（较慢但不占GPU显存）

        expert_start/expert_end: 该节点负责的专家范围 [start, end)
        """
        self.expert_start = expert_start
        self.expert_end = expert_end
        self.num_my_experts = expert_end - expert_start

        # 环境配置：优化显存分配碎片，限制内存缓存
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        self.experts_gpu = nn.ModuleDict()
        self.experts_cpu = {}

        # 动态计算GPU容量（预留25%显存）
        self.gpu_capacity = self._calculate_gpu_capacity(config)

        # 并发锁：保护CPU专家临时调入GPU的过程（max_concurrency=4时需要）
        self.gpu_lock = threading.Lock()

        print(f"\n[边缘网元] 智能内存管理模式启动 (并发度=4)")
        print(f"负责专家范围: {expert_start}-{expert_end-1} (共{self.num_my_experts}个)")
        print(f"配置：GPU({self.gpu_capacity}专家) + CPU({self.num_my_experts - self.gpu_capacity}专家)")
        
        # 初始清理
        torch.cuda.empty_cache()
        gc.collect()

        for eid in range(expert_start, expert_end):
            expert_file = os.path.join(split_weights_path, "experts", f"expert_{eid}.pt")
            if not os.path.exists(expert_file):
                continue

            # 1. 加载专家文件到 CPU (临时占用约 414MB RAM)
            expert_sd = torch.load(expert_file, map_location="cpu", weights_only=True)
            
            # 判断当前专家应该加载到GPU还是CPU（使用相对索引）
            load_to_gpu = ((eid - expert_start) < self.gpu_capacity)
            
            for l_idx in range(config.num_hidden_layers):
                # 2. 初始化该层 MLP（先在CPU上创建）
                mlp = Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size).bfloat16()
                
                prefix = f"model.layers.{l_idx}.mlp.experts.{eid}."
                # 提取层权重并立即转换为字典引用
                layer_sd = {k.replace(prefix, ""): v.bfloat16() for k, v in expert_sd.items() if k.startswith(prefix)}
                
                if layer_sd:
                    # 3. 核心优化：原地赋值，不产生重复内存占用
                    mlp.load_state_dict(layer_sd, assign=True)
                    mlp.eval()
                    
                    key = f"L{l_idx}_E{eid}"
                    if load_to_gpu:
                        # 移动到显存（逐层移动，避免瞬时显存峰值）
                        self.experts_gpu[key] = mlp.cuda()
                        # 立即清理临时变量，释放CPU内存
                        del layer_sd
                        # 每加载5层到GPU就清理一次显存碎片
                        if l_idx % 5 == 0:
                            torch.cuda.empty_cache()
                    else:
                        # 留在CPU内存
                        self.experts_cpu[key] = mlp.cpu()
                        del layer_sd
                
                # 每层处理完，手动清理层临时变量
                del mlp

            # 4. 重要：处理完一个专家（24层），彻底销毁文件字典并回收内存
            del expert_sd
            
            # 5. 强制回收：每个专家加载完都回收，确保内存稳定
            gc.collect()
            if load_to_gpu:
                torch.cuda.empty_cache()
            
            # 6. 额外的内存清理：每5个专家强制垃圾回收
            if eid % 5 == 0:
                gc.collect()
                gc.collect()  # 双重回收，清理循环引用
            
            if eid % 5 == 0 or eid == self.gpu_capacity - 1 + expert_start or eid == expert_end - 1:  # 加上最后一个
                # 实时监控系统内存占用 (RAM)
                import psutil
                ram_usage = psutil.Process().memory_info().rss / 1024**3
                vram_usage = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"进度: {eid - expert_start + 1}/{self.num_my_experts} (全局eid={eid}) | RAM: {ram_usage:.2f}GB | VRAM: {vram_usage:.2f}/{vram_total:.2f}GB ({vram_usage/vram_total*100:.1f}%)")
                
            # 关键检查点：刚完成GPU专家加载时的状态
            if eid == self.gpu_capacity - 1 + expert_start:
                print(f"\n>>> GPU专家加载完成检查点 <<<")
                vram_usage = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"已加载 {self.gpu_capacity} 个专家到GPU")
                print(f"显存占用: {vram_usage:.2f}GB / {vram_total:.2f}GB ({vram_usage/vram_total*100:.1f}%)")
                print(f"剩余专家将加载到CPU内存\n")

        print(f"\n[边缘网元] 部署成功！")
        print(f"最终状态：显存权重 {len(self.experts_gpu)} 项, 内存权重 {len(self.experts_cpu)} 项")
        
        # 深度内存清理
        print("🧹 执行深度内存清理...")
        gc.collect()
        gc.collect()
        gc.collect()  # 三重回收，清理所有临时对象
        torch.cuda.empty_cache()
        
        # 最终显存统计
        import psutil
        final_vram = torch.cuda.memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        final_ram = psutil.Process().memory_info().rss / 1024**3
        print(f"显存占用: {final_vram:.2f}GB / {total_vram:.2f}GB ({final_vram/total_vram*100:.1f}%)")
        print(f"内存占用: {final_ram:.2f}GB")

    def _calculate_gpu_capacity(self, config):
        """
        动态计算可以加载到GPU的专家数量（预留25%显存用于推理）
        
        策略：
        1. 获取总显存并预留25%用于推理（KV cache、激活值、base model开销）
        2. 估算单个专家占用的显存
        3. 计算能放入GPU的专家数量
        
        显存分配：75%加载专家模型，25%预留推理
        """
        # 获取显卡总显存（单位：GB）
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到显卡总显存: {total_vram_gb:.2f}GB")
        
        # 预留25%显存用于推理计算（KV cache、激活值、base model开销）
        # 使用75%加载专家模型，25%预留推理
        available_vram_gb = total_vram_gb * 0.65  # 65%用于专家模型
        print(f"可用显存(预留35%用于推理): {available_vram_gb:.2f}GB")
        
        # 估算单个专家的显存占用
        # Qwen1.5-MoE-2.7B 每个专家约为：
        # - 3层权重 (gate_proj, up_proj, down_proj)
        # - hidden_size=2048, intermediate_size=1408 (config.moe_intermediate_size)
        # - bfloat16 = 2 bytes/param
        hidden_size = config.hidden_size  # 2048
        intermediate_size = config.moe_intermediate_size  # 1408
        num_layers = config.num_hidden_layers  # 24层
        
        # 每个专家每层的参数量
        params_per_expert_per_layer = (
            hidden_size * intermediate_size +  # gate_proj
            hidden_size * intermediate_size +  # up_proj
            intermediate_size * hidden_size     # down_proj
        )
        
        # 单个专家（跨所有层）的参数量
        params_per_expert_total = params_per_expert_per_layer * num_layers
        
        # 转换为显存占用（bfloat16 = 2 bytes）
        bytes_per_expert = params_per_expert_total * 2
        gb_per_expert = bytes_per_expert / 1024**3
        
        print(f"单个专家显存占用: {gb_per_expert:.3f}GB")
        print(f"单个专家参数量: {params_per_expert_total/1e6:.2f}M")
        
        # 计算能放入的专家数量
        gpu_capacity = int(available_vram_gb / gb_per_expert)
        
        # 安全限制：至少1个，最多 num_my_experts-1 个（留一个在CPU避免极端情况）
        gpu_capacity = max(1, min(gpu_capacity, self.num_my_experts - 1))
        
        print(f"计算结果：可加载 {gpu_capacity} 个专家到GPU")
        
        return gpu_capacity

    def compute_expert(self, layer_idx, expert_id, tokens, weights):
        """
        混合计算策略：
        - GPU专家(20个)：在GPU上高速计算，真正并发
        - CPU专家(40个)：在CPU上计算，避免GPU显存占用，支持并发
        
        性能：GPU专家快，CPU专家慢10-20倍，但整体稳定不OOM
        """
        key = f"L{layer_idx}_E{expert_id}"

        with torch.no_grad():
            if key in self.experts_gpu:
                # GPU 专家：在GPU上高速计算（专家已在显存中）
                tokens_cuda = tokens.cuda().bfloat16()
                weights_cuda = weights.cuda().bfloat16()
                res = self.experts_gpu[key](tokens_cuda) * weights_cuda
                return res.cpu()
                
            elif key in self.experts_cpu:
                # CPU 专家：直接在CPU上计算（避免GPU显存占用）
                # 不调入GPU，完全在CPU上运行，支持真正并发
                tokens_cpu = tokens.cpu().bfloat16()
                weights_cpu = weights.cpu().bfloat16()
                
                # CPU上计算（较慢但不占用GPU显存）
                res = self.experts_cpu[key](tokens_cpu) * weights_cpu
                
                return res
                
            else:
                # 专家不存在（理论上不应该发生）
                return torch.zeros_like(tokens)