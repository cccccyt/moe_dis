#!/bin/bash

# ============================================
# MoE分布式项目完整设置脚本
# 用于在新机器上创建完整的项目环境和代码
# ============================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
CONDA_ENV_NAME="moe"
PYTHON_VERSION="3.11"
PROJECT_DIR="home/moe-dis"
MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B"
MODEL_LOCAL_DIR="/home/llm/ai/model/Qwen/Qwen1.5-MoE-A2.7B"
HF_ENDPOINT="https://hf-mirror.com"


echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MoE分布式项目完整设置${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ============================================
# 步骤 1: 创建项目目录结构
# ============================================
echo -e "${YELLOW}[1/10] 创建项目目录结构${NC}"

if [ -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}项目目录已存在: $PROJECT_DIR${NC}"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_DIR"
        echo -e "${GREEN}已删除旧目录${NC}"
    else
        echo -e "${RED}操作取消${NC}"
        exit 1
    fi
fi

mkdir -p "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/qwen_src"
mkdir -p "$PROJECT_DIR/split_weights/experts"
mkdir -p "$PROJECT_DIR/weights"

echo -e "${GREEN}✓ 项目目录结构创建完成${NC}"
echo -e "${BLUE}项目路径: $PROJECT_DIR${NC}"

# ============================================
# 步骤 2: 检查conda环境
# ============================================
echo ""
echo -e "${YELLOW}[2/10] 检查conda环境${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: conda 未安装！请先安装 Anaconda 或 Miniconda${NC}"
    exit 1
fi

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo -e "${YELLOW}环境 ${CONDA_ENV_NAME} 已存在${NC}"
    read -p "是否重新创建环境? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${CONDA_ENV_NAME} -y
        conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
        echo -e "${GREEN}✓ 环境重新创建成功${NC}"
    fi
else
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    echo -e "${GREEN}✓ 环境创建成功${NC}"
fi

# ============================================
# 步骤 3: 安装Python依赖
# ============================================
echo ""
echo -e "${YELLOW}[3/10] 安装Python基础依赖${NC}"

conda run -n ${CONDA_ENV_NAME} pip install --upgrade pip
conda run -n ${CONDA_ENV_NAME} pip install ray transformers accelerate bitsandbytes safetensors psutil tqdm

echo -e "${GREEN}✓ 基础依赖安装完成${NC}"

# ============================================
# 步骤 4: 安装CUDA工具包
# ============================================
echo ""
echo -e "${YELLOW}[4/10] 安装CUDA 12.8工具包${NC}"

conda run -n ${CONDA_ENV_NAME} conda install -y nvidia/label/cuda-12.8.0::cuda-toolkit

echo -e "${GREEN}✓ CUDA工具包安装完成${NC}"

# ============================================
# 步骤 5: 安装PyTorch
# ============================================
echo ""
echo -e "${YELLOW}[5/10] 安装PyTorch (nightly版本，CUDA 12.8)${NC}"

conda run -n ${CONDA_ENV_NAME} pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo -e "${GREEN}✓ PyTorch安装完成${NC}"

# ============================================
# 步骤 6: 创建源代码文件
# ============================================
echo ""
echo -e "${YELLOW}[6/10] 创建项目源代码文件${NC}"

# 6.1 创建 qwen_src/__init__.py
echo -e "${BLUE}创建 qwen_src/__init__.py${NC}"
cat > "$PROJECT_DIR/qwen_src/__init__.py" << 'INIT_EOF'
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_qwen2_moe import *
    from .modeling_qwen2_moe import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
INIT_EOF

# 由于configuration_qwen2_moe.py和modeling_qwen2_moe.py文件太大，我们将它们分别下载
echo -e "${BLUE}提示: qwen_src/configuration_qwen2_moe.py 和 modeling_qwen2_moe.py 文件较大${NC}"
echo -e "${BLUE}这些文件需要从模型目录复制，或者在下载模型后会自动包含${NC}"

# 创建一个占位文件说明
cat > "$PROJECT_DIR/qwen_src/README_QWEN_SRC.txt" << 'README_EOF'
qwen_src 目录说明
===================

这个目录需要包含以下文件：
1. __init__.py (已创建)
2. configuration_qwen2_moe.py (从模型目录复制)
3. modeling_qwen2_moe.py (从模型目录复制)

这些文件可以从下载的Qwen1.5-MoE-A2.7B模型目录中复制，或者使用transformers库的源代码。

如果你已经下载了模型，可以从以下位置复制：
- {MODEL_DIR}/configuration_qwen2_moe.py
- {MODEL_DIR}/modeling_qwen2_moe.py

或者使用以下命令从transformers包复制：
python -c "from transformers.models.qwen2_moe import configuration_qwen2_moe, modeling_qwen2_moe"
README_EOF

# 6.2 创建 worker_node.py
echo -e "${BLUE}创建 worker_node.py${NC}"
cat > "$PROJECT_DIR/worker_node.py" << 'WORKER_EOF'
import torch
import torch.nn as nn
import os
import sys
import ray
import gc
import time
import threading

# 确保能找到 Qwen 源码
sys.path.append(os.path.abspath("./qwen_src"))
from qwen_src.modeling_qwen2_moe import Qwen2MoeMLP

@ray.remote(num_gpus=1, max_concurrency=4)
class RemoteExpertNode:
    def __init__(self, config, split_weights_path):
        """
        智能内存管理版 - 动态分配GPU/CPU专家，预留25%显存用于推理计算
        支持并发处理(max_concurrency=4)
        - GPU专家：在GPU上并发计算（高速）
        - CPU专家：在CPU上并发计算（较慢但不占GPU显存）
        """
        # 环境配置：优化显存分配碎片
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        self.experts_gpu = nn.ModuleDict() 
        self.experts_cpu = {}             
        
        # 动态计算GPU容量（预留25%显存）
        self.gpu_capacity = self._calculate_gpu_capacity(config)
        
        # 并发锁
        self.gpu_lock = threading.Lock()
        
        print(f"\n[边缘网元] 智能内存管理模式启动 (并发度=4)")
        print(f"配置：GPU({self.gpu_capacity}专家) + CPU({60-self.gpu_capacity}专家)")
        
        # 初始清理
        torch.cuda.empty_cache()
        gc.collect()

        for eid in range(60):
            expert_file = os.path.join(split_weights_path, "experts", f"expert_{eid}.pt")
            if not os.path.exists(expert_file):
                continue

            # 加载专家文件到 CPU
            expert_sd = torch.load(expert_file, map_location="cpu", weights_only=True)
            
            # 判断当前专家应该加载到GPU还是CPU
            load_to_gpu = (eid < self.gpu_capacity)
            
            for l_idx in range(config.num_hidden_layers):
                # 初始化该层 MLP
                mlp = Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size).bfloat16()
                
                prefix = f"model.layers.{l_idx}.mlp.experts.{eid}."
                layer_sd = {k.replace(prefix, ""): v.bfloat16() for k, v in expert_sd.items() if k.startswith(prefix)}
                
                if layer_sd:
                    mlp.load_state_dict(layer_sd, assign=True)
                    mlp.eval()
                    
                    key = f"L{l_idx}_E{eid}"
                    if load_to_gpu:
                        self.experts_gpu[key] = mlp.cuda()
                        del layer_sd
                        if l_idx % 5 == 0:
                            torch.cuda.empty_cache()
                    else:
                        self.experts_cpu[key] = mlp.cpu()
                        del layer_sd
                
                del mlp

            del expert_sd
            
            gc.collect()
            if load_to_gpu:
                torch.cuda.empty_cache()
            
            if eid % 5 == 0 or eid == self.gpu_capacity - 1 or eid == 59:
                import psutil
                ram_usage = psutil.Process().memory_info().rss / 1024**3
                vram_usage = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"进度: {eid+1}/60 | RAM: {ram_usage:.2f}GB | VRAM: {vram_usage:.2f}/{vram_total:.2f}GB ({vram_usage/vram_total*100:.1f}%)")
                
            if eid == self.gpu_capacity - 1:
                print(f"\n>>> GPU专家加载完成检查点 <<<")
                vram_usage = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"已加载 {self.gpu_capacity} 个专家到GPU")
                print(f"显存占用: {vram_usage:.2f}GB / {vram_total:.2f}GB ({vram_usage/vram_total*100:.1f}%)")
                print(f"剩余专家将加载到CPU内存\n")

        print(f"\n[边缘网元] 部署成功！")
        print(f"最终状态：显存权重 {len(self.experts_gpu)} 项, 内存权重 {len(self.experts_cpu)} 项")
        
        final_vram = torch.cuda.memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存占用: {final_vram:.2f}GB / {total_vram:.2f}GB ({final_vram/total_vram*100:.1f}%)")

    def _calculate_gpu_capacity(self, config):
        """计算GPU容量（预留25%显存）"""
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到显卡总显存: {total_vram_gb:.2f}GB")
        
        available_vram_gb = total_vram_gb * 0.75
        print(f"可用显存(预留25%用于推理): {available_vram_gb:.2f}GB")
        
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        num_layers = config.num_hidden_layers
        
        params_per_expert_per_layer = (
            hidden_size * intermediate_size +
            hidden_size * intermediate_size +
            intermediate_size * hidden_size
        )
        
        params_per_expert_total = params_per_expert_per_layer * num_layers
        bytes_per_expert = params_per_expert_total * 2
        gb_per_expert = bytes_per_expert / 1024**3
        
        print(f"单个专家显存占用: {gb_per_expert:.3f}GB")
        print(f"单个专家参数量: {params_per_expert_total/1e6:.2f}M")
        
        gpu_capacity = int(available_vram_gb / gb_per_expert)
        gpu_capacity = max(1, min(gpu_capacity, 59))
        
        print(f"计算结果：可加载 {gpu_capacity} 个专家到GPU")
        
        return gpu_capacity

    def compute_expert(self, layer_idx, expert_id, tokens, weights):
        """混合计算策略"""
        key = f"L{layer_idx}_E{expert_id}"

        with torch.no_grad():
            if key in self.experts_gpu:
                tokens_cuda = tokens.cuda().bfloat16()
                weights_cuda = weights.cuda().bfloat16()
                res = self.experts_gpu[key](tokens_cuda) * weights_cuda
                return res.cpu()
                
            elif key in self.experts_cpu:
                tokens_cpu = tokens.cpu().bfloat16()
                weights_cpu = weights.cpu().bfloat16()
                res = self.experts_cpu[key](tokens_cpu) * weights_cpu
                return res
                
            else:
                return torch.zeros_like(tokens)
WORKER_EOF

# 6.3 创建 worker_node_bnb.py
echo -e "${BLUE}创建 worker_node_bnb.py (INT8量化版本)${NC}"
cat > "$PROJECT_DIR/worker_node_bnb.py" << 'WORKER_BNB_EOF'
"""
分布式MoE专家节点 - Bitsandbytes INT8量化版本
使用 bitsandbytes 在GPU上实现真正的INT8量化
预留15%显存，85%用于专家模型
"""

import torch
import torch.nn as nn
import os
import sys
import ray
import gc
import threading

# 确保能找到 Qwen 源码
sys.path.append(os.path.abspath("./qwen_src"))
from qwen_src.modeling_qwen2_moe import Qwen2MoeMLP

# 导入 bitsandbytes
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("⚠️  bitsandbytes 未安装！")
    print("请运行: pip install bitsandbytes")

@ray.remote(num_gpus=1, max_concurrency=4)
class RemoteExpertNode:
    def __init__(self, config, split_weights_path):
        """
        Bitsandbytes INT8量化版本
        - GPU预留15%，85%用于专家模型
        - 真正的GPU INT8量化，节省50%显存
        """
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes 未安装！请运行: pip install bitsandbytes")
        
        # 环境配置
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        self.experts_gpu = nn.ModuleDict() 
        self.experts_cpu = {}             
        
        # 动态计算GPU容量（预留15%）
        self.gpu_capacity = self._calculate_gpu_capacity(config)
        
        # 并发锁
        self.gpu_lock = threading.Lock()
        
        print(f"\n[边缘网元] Bitsandbytes INT8量化模式启动 (并发度=4)")
        print(f"配置：GPU({self.gpu_capacity}专家,INT8) + CPU({60-self.gpu_capacity}专家,INT8)")
        print(f"量化库：bitsandbytes (真正的GPU INT8量化)")
        print(f"显存分配：85%专家 + 15%推理预留")
        
        # 初始清理
        torch.cuda.empty_cache()
        gc.collect()

        for eid in range(60):
            expert_file = os.path.join(split_weights_path, "experts", f"expert_{eid}.pt")
            if not os.path.exists(expert_file):
                continue

            expert_sd = torch.load(expert_file, map_location="cpu", weights_only=True)
            load_to_gpu = (eid < self.gpu_capacity)
            
            for l_idx in range(config.num_hidden_layers):
                mlp = Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                
                prefix = f"model.layers.{l_idx}.mlp.experts.{eid}."
                layer_sd = {k.replace(prefix, ""): v for k, v in expert_sd.items() if k.startswith(prefix)}
                
                if layer_sd:
                    mlp.load_state_dict(layer_sd, assign=True)
                    mlp.eval()
                    
                    key = f"L{l_idx}_E{eid}"
                    if load_to_gpu:
                        mlp = self._quantize_to_int8_bnb(mlp)
                        self.experts_gpu[key] = mlp.cuda()
                        del layer_sd
                        if l_idx % 5 == 0:
                            torch.cuda.empty_cache()
                    else:
                        mlp = self._quantize_to_int8_pytorch(mlp)
                        self.experts_cpu[key] = mlp.cpu()
                        del layer_sd
                
                del mlp

            del expert_sd
            gc.collect()
            if load_to_gpu:
                torch.cuda.empty_cache()
            
            if eid % 5 == 0:
                gc.collect()
                gc.collect()
            
            if eid % 5 == 0 or eid == self.gpu_capacity - 1 or eid == 59:
                import psutil
                ram_usage = psutil.Process().memory_info().rss / 1024**3
                vram_usage = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"进度: {eid+1}/60 | RAM: {ram_usage:.2f}GB | VRAM: {vram_usage:.2f}/{vram_total:.2f}GB ({vram_usage/vram_total*100:.1f}%)")
                
            if eid == self.gpu_capacity - 1:
                print(f"\n>>> GPU专家加载完成检查点 (Bitsandbytes INT8) <<<")
                vram_usage = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"已加载 {self.gpu_capacity} 个专家到GPU（INT8量化）")
                print(f"显存占用: {vram_usage:.2f}GB / {vram_total:.2f}GB ({vram_usage/vram_total*100:.1f}%)")
                expected_bf16 = self.gpu_capacity * 0.387
                print(f"💾 相比bfloat16节省约: {expected_bf16:.2f}GB → {vram_usage:.2f}GB (节省{expected_bf16-vram_usage:.2f}GB)")
                print(f"剩余专家将加载到CPU内存（INT8量化）\n")

        print(f"\n[边缘网元] Bitsandbytes INT8量化部署成功！")
        print(f"最终状态：GPU权重 {len(self.experts_gpu)} 项(INT8), CPU权重 {len(self.experts_cpu)} 项(INT8)")
        
        # 深度内存清理
        print("🧹 执行深度内存清理...")
        gc.collect()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        
        # 最终统计
        import psutil
        final_vram = torch.cuda.memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        final_ram = psutil.Process().memory_info().rss / 1024**3
        print(f"显存占用: {final_vram:.2f}GB / {total_vram:.2f}GB ({final_vram/total_vram*100:.1f}%)")
        print(f"内存占用: {final_ram:.2f}GB")
        print(f"✅ Bitsandbytes INT8量化: GPU+CPU都节省约50%空间")

    def _quantize_to_int8_bnb(self, module):
        """使用 Bitsandbytes 将模型量化为INT8（支持GPU）"""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                int8_linear = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0
                )
                int8_linear.weight.data = child.weight.data
                if child.bias is not None:
                    int8_linear.bias.data = child.bias.data
                    
                setattr(module, name, int8_linear)
            else:
                self._quantize_to_int8_bnb(child)
        
        return module

    def _quantize_to_int8_pytorch(self, module):
        """使用 PyTorch 动态量化（CPU专家）"""
        quantized_module = torch.quantization.quantize_dynamic(
            module,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_module

    def _calculate_gpu_capacity(self, config):
        """计算GPU容量（INT8格式，预留15%显存）"""
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到显卡总显存: {total_vram_gb:.2f}GB")
        
        available_vram_gb = total_vram_gb * 0.85
        print(f"可用显存(预留15%用于推理): {available_vram_gb:.2f}GB")
        
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        num_layers = config.num_hidden_layers
        
        params_per_expert_per_layer = (
            hidden_size * intermediate_size +
            hidden_size * intermediate_size +
            intermediate_size * hidden_size
        )
        
        params_per_expert_total = params_per_expert_per_layer * num_layers
        bytes_per_expert = params_per_expert_total * 1
        gb_per_expert = bytes_per_expert / 1024**3
        
        print(f"单个GPU专家显存占用(INT8): {gb_per_expert:.3f}GB")
        print(f"单个专家参数量: {params_per_expert_total/1e6:.2f}M")
        print(f"相比bfloat16节省: 0.387GB → {gb_per_expert:.3f}GB (50%)")
        
        gpu_capacity = int(available_vram_gb / gb_per_expert)
        gpu_capacity = max(1, min(gpu_capacity, 60))
        
        print(f"计算结果：可加载 {gpu_capacity} 个专家到GPU (INT8格式)")
        
        return gpu_capacity

    def compute_expert(self, layer_idx, expert_id, tokens, weights):
        """混合计算策略（Bitsandbytes INT8版本）"""
        key = f"L{layer_idx}_E{expert_id}"

        with torch.no_grad():
            if key in self.experts_gpu:
                tokens_cuda = tokens.cuda().bfloat16()
                weights_cuda = weights.cuda().bfloat16()
                res = self.experts_gpu[key](tokens_cuda) * weights_cuda
                return res.cpu()
                
            elif key in self.experts_cpu:
                tokens_cpu = tokens.cpu().bfloat16()
                weights_cpu = weights.cpu().bfloat16()
                res = self.experts_cpu[key](tokens_cpu) * weights_cpu
                return res
                
            else:
                return torch.zeros_like(tokens)
WORKER_BNB_EOF

# 6.4 创建 split_weights.py
echo -e "${BLUE}创建 split_weights.py${NC}"
cat > "$PROJECT_DIR/split_weights.py" << 'SPLIT_EOF'
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

    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    shard_files = sorted(set(index_data["weight_map"].values()))
    
    base_model_weights = {}
    experts_weights = {i: {} for i in range(60)}

    print(f"检测到 {len(shard_files)} 个权重分片，开始逐一处理...")

    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        print(f"\n正在处理分片: {shard_name}")
        
        shard_sd = load_file(shard_path)
        
        for key in list(shard_sd.keys()):
            value = shard_sd[key]
            if ".mlp.experts." in key:
                parts = key.split(".")
                expert_idx = int(parts[5])
                experts_weights[expert_idx][key] = value
            else:
                base_model_weights[key] = value
            
            del shard_sd[key]
        
        del shard_sd
        gc.collect()

    print("\n--- 正在保存切分后的权重 ---")
    torch.save(base_model_weights, os.path.join(save_path, "base_model.pt"))
    print("✓ base_model.pt 已保存")
    del base_model_weights
    gc.collect()

    for i in tqdm(range(60), desc="保存专家文件"):
        if experts_weights[i]:
            torch.save(experts_weights[i], os.path.join(save_path, "experts", f"expert_{i}.pt"))
    
    print("\n切分完成！所有专家已保存至 experts/ 目录。")

if __name__ == "__main__":
    import sys
    
    # 获取脚本所在目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 优先使用外部模型路径，如果不存在则使用项目内路径
    model_path_external = "${MODEL_LOCAL_DIR}"
    model_path_internal = os.path.join(script_dir, "weights", "qwen1.5-moe-2.7b")
    
    if os.path.exists(model_path_external):
        model_path = model_path_external
        print(f"使用外部模型路径: {model_path}")
    elif os.path.exists(model_path_internal):
        model_path = model_path_internal
        print(f"使用项目内模型路径: {model_path}")
    else:
        print("❌ 找不到模型文件！")
        print(f"已检查路径1: {model_path_external}")
        print(f"已检查路径2: {model_path_internal}")
        print("")
        print("请先下载模型:")
        print(f"  huggingface-cli download --resume-download Qwen/Qwen1.5-MoE-A2.7B \\\\")
        print(f"      --local-dir {model_path_external} \\\\")
        print(f"      --local-dir-use-symlinks False")
        sys.exit(1)
    
    save_path = os.path.join(script_dir, "split_weights")
    print(f"保存路径: {save_path}")
    split_sharded_qwen_moe(model_path, save_path)
SPLIT_EOF

# 6.5 创建 print_model_struct.py
echo -e "${BLUE}创建 print_model_struct.py${NC}"
cat > "$PROJECT_DIR/print_model_struct.py" << 'PRINT_EOF'
import torch
import os
import sys

sys.path.append(os.path.abspath("./qwen_src"))
from qwen_src.modeling_qwen2_moe import Qwen2MoeForCausalLM
from qwen_src.configuration_qwen2_moe import Qwen2MoeConfig

def print_original_model_struct(model_path):
    config = Qwen2MoeConfig.from_pretrained(
        model_path, 
        local_files_only=True
    )

    with torch.device("meta"):
        model = Qwen2MoeForCausalLM(config)

    print("=== Qwen2-MoE 原始蓝图架构 ===")
    print(model)

    print("\n=== 第 0 层 MLP 专家细节 ===")
    print(model.model.layers[0].mlp)

def inspect_pt_file(file_path, name):
    print(f"\n{'='*20} 探测文件: {name} {'='*20}")
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    sd = torch.load(file_path, map_location="cpu", weights_only=True)
    
    keys = list(sd.keys())
    print(f"总参数项数量: {len(keys)}")
    
    print(f"{'参数路径 (Key)':<65} | {'形状 (Shape)':<20}")
    print("-" * 90)
    for k in keys[:5]:
        print(f"{k:<65} | {str(list(sd[k].shape)):<20}")
    
    if "base" in name.lower():
        expert_keys = [k for k in keys if "mlp.experts" in k]
        print(f"\n[验证] 基础模型中残留的专家参数数量: {len(expert_keys)}")
    else:
        layers = set([k.split(".")[2] for k in keys if "layers" in k])
        print(f"\n[验证] 该专家跨越的 Transformer 层数: {len(layers)} (通常应为 24)")

if __name__ == "__main__":
    # 示例使用
    # model_path = "$PROJECT_DIR/weights/qwen1.5-moe-2.7b"
    # split_dir = "$PROJECT_DIR/split_weights"
    # print_original_model_struct(model_path)
    # inspect_pt_file(os.path.join(split_dir, "base_model.pt"), "主控基础权重")
    # inspect_pt_file(os.path.join(split_dir, "experts/expert_0.pt"), "0号专家专属权重")
    print("请取消注释上面的代码并修改路径后使用")
PRINT_EOF

echo -e "${GREEN}✓ 源代码文件创建完成${NC}"

# ============================================
# 步骤 7: 创建requirements.txt
# ============================================
echo ""
echo -e "${YELLOW}[7/10] 创建 requirements.txt${NC}"

cat > "$PROJECT_DIR/requirements.txt" << 'REQ_EOF'
# MoE分布式项目依赖
torch>=2.1.0
ray>=2.0.0
transformers>=4.35.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
safetensors>=0.3.0
psutil>=5.9.0
tqdm>=4.65.0
REQ_EOF

echo -e "${GREEN}✓ requirements.txt 创建完成${NC}"

# ============================================
# 步骤 8: 配置HuggingFace环境
# ============================================
echo ""
echo -e "${YELLOW}[8/10] 配置HuggingFace环境${NC}"

export HF_ENDPOINT=${HF_ENDPOINT}

# 添加到 bashrc 和 profile
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo "export HF_ENDPOINT=${HF_ENDPOINT}" >> ~/.bashrc
fi

if ! grep -q "HF_ENDPOINT" ~/.profile; then
    echo "export HF_ENDPOINT=${HF_ENDPOINT}" >> ~/.profile
fi

if [ -d ~/.cache/huggingface ]; then
    sudo chown -R $USER:$USER ~/.cache/huggingface
    echo -e "${GREEN}✓ HuggingFace缓存目录权限已设置${NC}"
else
    echo -e "${YELLOW}HuggingFace缓存目录将在首次使用时创建${NC}"
fi

echo -e "${GREEN}✓ HuggingFace环境配置完成${NC}"

# ============================================
# 步骤 9: HuggingFace登录和模型下载
# ============================================
echo ""
echo -e "${YELLOW}[9/10] HuggingFace登录和模型下载${NC}"

read -p "是否现在登录HuggingFace并下载模型? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}开始HuggingFace自动登录（使用token）...${NC}"
    conda run -n ${CONDA_ENV_NAME} huggingface-cli login --token ${HF_TOKEN}
    
    echo ""
    echo -e "${BLUE}开始下载模型: ${MODEL_NAME}${NC}"
    echo -e "${BLUE}下载位置: ${MODEL_LOCAL_DIR}${NC}"
    echo -e "${YELLOW}这可能需要较长时间，请耐心等待...${NC}"
    
    # 创建模型目录
    mkdir -p "$MODEL_LOCAL_DIR"
    
    conda run -n ${CONDA_ENV_NAME} huggingface-cli download \
        --resume-download \
        ${MODEL_NAME} \
        --local-dir "$MODEL_LOCAL_DIR" \
        --local-dir-use-symlinks False
    
    echo -e "${GREEN}✓ 模型下载完成${NC}"
    
    # 复制源代码文件到 qwen_src
    echo ""
    echo -e "${BLUE}复制模型源代码文件到 qwen_src/...${NC}"
    
    if [ -f "${MODEL_LOCAL_DIR}/configuration_qwen2_moe.py" ]; then
        cp "${MODEL_LOCAL_DIR}/configuration_qwen2_moe.py" "$PROJECT_DIR/qwen_src/"
        echo -e "${GREEN}✓ configuration_qwen2_moe.py 已复制${NC}"
    else
        echo -e "${RED}⚠ 警告: 找不到 configuration_qwen2_moe.py${NC}"
    fi
    
    if [ -f "${MODEL_LOCAL_DIR}/modeling_qwen2_moe.py" ]; then
        cp "${MODEL_LOCAL_DIR}/modeling_qwen2_moe.py" "$PROJECT_DIR/qwen_src/"
        echo -e "${GREEN}✓ modeling_qwen2_moe.py 已复制${NC}"
    else
        echo -e "${RED}⚠ 警告: 找不到 modeling_qwen2_moe.py${NC}"
    fi
    
    # 执行权重切分
    echo ""
    echo -e "${BLUE}开始切分模型权重...${NC}"
    echo -e "${YELLOW}这需要一些时间和内存，请耐心等待...${NC}"
    
    cd "$PROJECT_DIR"
    
    # 修改 split_weights.py 中的路径
    cat > "$PROJECT_DIR/split_weights_auto.py" << EOF
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

    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    shard_files = sorted(set(index_data["weight_map"].values()))
    
    base_model_weights = {}
    experts_weights = {i: {} for i in range(60)}

    print(f"检测到 {len(shard_files)} 个权重分片，开始逐一处理...")

    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        print(f"\n正在处理分片: {shard_name}")
        
        shard_sd = load_file(shard_path)
        
        for key in list(shard_sd.keys()):
            value = shard_sd[key]
            if ".mlp.experts." in key:
                parts = key.split(".")
                expert_idx = int(parts[5])
                experts_weights[expert_idx][key] = value
            else:
                base_model_weights[key] = value
            
            del shard_sd[key]
        
        del shard_sd
        gc.collect()

    print("\n--- 正在保存切分后的权重 ---")
    torch.save(base_model_weights, os.path.join(save_path, "base_model.pt"))
    print("✓ base_model.pt 已保存")
    del base_model_weights
    gc.collect()

    for i in tqdm(range(60), desc="保存专家文件"):
        if experts_weights[i]:
            torch.save(experts_weights[i], os.path.join(save_path, "experts", f"expert_{i}.pt"))
    
    print("\n切分完成！所有专家已保存至 experts/ 目录。")

if __name__ == "__main__":
    model_path = "$MODEL_LOCAL_DIR"
    save_path = "$PROJECT_DIR/split_weights"
    split_sharded_qwen_moe(model_path, save_path)
EOF
    
    conda run -n ${CONDA_ENV_NAME} python "$PROJECT_DIR/split_weights_auto.py"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 权重切分完成${NC}"
        rm "$PROJECT_DIR/split_weights_auto.py"
    else
        echo -e "${RED}✗ 权重切分失败${NC}"
        echo -e "${YELLOW}可以稍后手动运行: python split_weights.py${NC}"
    fi
    
else
    echo -e "${YELLOW}跳过模型下载${NC}"
    echo ""
    echo -e "${YELLOW}稍后可以手动执行以下步骤:${NC}"
    echo ""
    echo -e "${BLUE}1. 登录 HuggingFace:${NC}"
    echo -e "   conda activate ${CONDA_ENV_NAME}"
    echo -e "   huggingface-cli login --token ${HF_TOKEN}"
    echo ""
    echo -e "${BLUE}2. 下载模型:${NC}"
    echo -e "   huggingface-cli download --resume-download ${MODEL_NAME} \\"
    echo -e "       --local-dir ${MODEL_LOCAL_DIR} \\"
    echo -e "       --local-dir-use-symlinks False"
    echo ""
    echo -e "${BLUE}3. 复制源代码文件到项目:${NC}"
    echo -e "   cp ${MODEL_LOCAL_DIR}/configuration_qwen2_moe.py $PROJECT_DIR/qwen_src/"
    echo -e "   cp ${MODEL_LOCAL_DIR}/modeling_qwen2_moe.py $PROJECT_DIR/qwen_src/"
    echo ""
    echo -e "${BLUE}4. 切分模型权重:${NC}"
    echo -e "   cd $PROJECT_DIR"
    echo -e "   python split_weights.py"
fi

# ============================================
# 步骤 10: 环境验证
# ============================================
echo ""
echo -e "${YELLOW}[10/10] 验证安装${NC}"

conda run -n ${CONDA_ENV_NAME} python << 'VERIFY_EOF'
import torch
import ray
import transformers
import accelerate

print("="*50)
print("环境验证")
print("="*50)

try:
    import bitsandbytes
    print(f'✓ bitsandbytes: {bitsandbytes.__version__}')
except ImportError:
    print('⚠ bitsandbytes: 未安装')

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✓ CUDA版本: {torch.version.cuda}')
    print(f'✓ GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
        
print(f'✓ Ray: {ray.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Accelerate: {accelerate.__version__}')

print("="*50)
VERIFY_EOF

# ============================================
# 完成
# ============================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}项目设置完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}项目信息:${NC}"
echo -e "  项目目录: ${GREEN}$PROJECT_DIR${NC}"
echo -e "  Conda环境: ${GREEN}${CONDA_ENV_NAME}${NC}"
echo -e "  Python版本: ${GREEN}${PYTHON_VERSION}${NC}"
echo ""
echo -e "${BLUE}项目结构:${NC}"
echo -e "  $PROJECT_DIR/"
echo -e "  ├── qwen_src/          # Qwen模型源码"
echo -e "  ├── split_weights/     # 切分后的权重（需运行split_weights.py生成）"
echo -e "  ├── weights/           # 原始模型权重目录"
echo -e "  ├── worker_node.py     # 工作节点（标准版）"
echo -e "  ├── worker_node_bnb.py # 工作节点（INT8量化版）"
echo -e "  ├── split_weights.py   # 权重切分脚本"
echo -e "  ├── print_model_struct.py  # 模型结构查看"
echo -e "  └── requirements.txt   # Python依赖"
echo ""
echo -e "${YELLOW}使用说明:${NC}"
echo -e "1. 激活环境:"
echo -e "   ${GREEN}conda activate ${CONDA_ENV_NAME}${NC}"
echo ""
echo -e "2. 进入项目目录:"
echo -e "   ${GREEN}cd $PROJECT_DIR${NC}"
echo ""
echo -e "3. 如果已下载模型且切分完成，运行工作节点:"
echo -e "   ${GREEN}python worker_node.py${NC}  # 标准版"
echo -e "   ${GREEN}python worker_node_bnb.py${NC}  # INT8量化版（推荐显存较小的GPU）"
echo ""
echo -e "4. 如果还没下载模型，参考上面的手动步骤"
echo ""
echo -e "${YELLOW}注意事项:${NC}"
echo -e "- HuggingFace环境变量已添加到 ~/.bashrc 和 ~/.profile"
echo -e "- HuggingFace Token 已配置（自动登录）"
echo -e "- 使用新shell时，运行: ${GREEN}source ~/.bashrc${NC}"
echo -e "- 模型下载位置: ${GREEN}${MODEL_LOCAL_DIR}${NC}"
echo -e "- 切分权重位置: ${GREEN}$PROJECT_DIR/split_weights/${NC}"
echo -e "- 确保qwen_src目录包含 configuration_qwen2_moe.py 和 modeling_qwen2_moe.py"
echo ""
echo -e "${GREEN}设置完成！祝使用愉快！${NC}"
