"""
分布式MoE推理服务 - API版本（多机扩展版）
提供HTTP REST API接口，支持其他程序调用

=== 三机部署指令 ===

机器1 (Master, IP 10.29.155.44):
  ray stop --force
  ray start --head --temp-dir=/home/oai/ray_tmp

机器2 (Worker 0):
  ray stop --force
  ray start --address='10.29.155.44:6379' --resources='{"worker_0": 1}' --num-gpus=1

机器3 (Worker 1):
  ray stop --force
  ray start --address='10.29.155.44:6379' --resources='{"worker_1": 1}' --num-gpus=1

端口清理:
  lsof -t -i :8000 | xargs kill -9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from transformers import AutoTokenizer

# 导入本地源码
sys.path.append(os.path.abspath("./qwen_src"))
from qwen_src.configuration_qwen2_moe import Qwen2MoeConfig
from qwen_src.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeSparseMoeBlock, Qwen2MoeMLP

# 导入从节点类
from worker_node import RemoteExpertNode

# ==========================================
# FastAPI 应用和数据模型
# ==========================================
app = FastAPI(
    title="分布式MoE推理服务",
    description="基于Ray的分布式Mixture-of-Experts推理API (UPF ↔ MEC)",
    version="1.0.0"
)

# 请求模型
class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="输入的问题或提示", example="什么是人工智能？")
    max_new_tokens: int = Field(50, ge=1, le=200, description="生成的最大token数")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="温度参数，控制随机性")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="nucleus sampling参数")
    do_sample: bool = Field(True, description="是否使用采样")

# 响应模型
class InferenceResponse(BaseModel):
    success: bool = Field(..., description="请求是否成功")
    result: str = Field(..., description="生成的文本结果")
    prompt: str = Field(..., description="原始输入")
    elapsed_time: float = Field(..., description="推理耗时（秒）")
    num_tokens: int = Field(..., description="生成的token数量")
    error: Optional[str] = Field(None, description="错误信息（如果有）")

# 健康检查响应
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_count: int
    device: str

# ==========================================
# 全局模型和配置（服务启动时初始化）
# ==========================================
model = None
tokenizer = None
master_device = None
config = None

# ==========================================
# Monkey Patch 补丁（与原版相同）
# ==========================================
def lean_moe_init(self, config):
    nn.Module.__init__(self)
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob
    self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    self.experts = nn.ModuleDict()
    self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
    self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
    self.expert_actor_map = None  # list of 60 actor refs (or None), indexed by global expert_id

def distributed_forward_patch(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    tokens_flat = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(tokens_flat)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros_like(tokens_flat)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    futures = []
    token_indices_map = []
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()
        row_idx, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() > 0:
            actor = self.expert_actor_map[expert_idx]
            if actor is not None:
                fut = actor.compute_expert.remote(
                    self.layer_idx, expert_idx, tokens_flat[token_idx], routing_weights[token_idx, row_idx, None]
                )
                futures.append(fut)
                token_indices_map.append(token_idx)

    all_results = ray.get(futures)
    for res_hidden, t_idx in zip(all_results, token_indices_map):
        final_hidden_states.index_add_(0, t_idx, res_hidden.to(tokens_flat.device))

    shared_output = self.shared_expert(tokens_flat)
    shared_output = torch.sigmoid(self.shared_expert_gate(tokens_flat)) * shared_output

    return (final_hidden_states + shared_output).reshape(batch_size, sequence_length, hidden_dim), router_logits

# ==========================================
# 模型初始化函数
# ==========================================

# ---- 多机专家分区配置 ----
NUM_EXPERT_WORKERS = 2  # 远程 expert worker 的数量（不含 master 本机）
# 自动将 60 个专家均分给各 worker
_experts_per_worker = 60 // NUM_EXPERT_WORKERS
EXPERT_RANGES = []
for _i in range(NUM_EXPERT_WORKERS):
    _s = _i * _experts_per_worker
    _e = _s + _experts_per_worker if _i < NUM_EXPERT_WORKERS - 1 else 60
    EXPERT_RANGES.append((_s, _e))
# 例如 NUM_EXPERT_WORKERS=2 时: [(0, 30), (30, 60)]
# 例如 NUM_EXPERT_WORKERS=3 时: [(0, 20), (20, 40), (40, 60)]


def initialize_model():
    """启动时初始化模型（只执行一次）"""
    global model, tokenizer, master_device, config

    print("\n" + "="*60)
    print("🚀 正在初始化分布式MoE推理服务（多机版）...")
    print(f"   专家分区: {EXPERT_RANGES}")
    print("="*60)

    # 连接Ray集群
    ray.init(address='auto')
    print("✅ 已连接到Ray集群")

    # 加载配置
    model_path = os.path.abspath("./weights/qwen1.5-moe-2.7b-chat")
    split_path = os.path.abspath("./split_weights_chat")
    config = Qwen2MoeConfig.from_pretrained(model_path)

    # 应用分布式补丁
    Qwen2MoeSparseMoeBlock.__init__ = lean_moe_init
    Qwen2MoeSparseMoeBlock.forward = distributed_forward_patch

    # ---- 创建远程专家Actor（每个 worker 负责一个专家范围）----
    expert_ranges_with_actors = []
    for worker_id, (start, end) in enumerate(EXPERT_RANGES):
        print(f"  创建 Worker {worker_id}: 负责专家 {start}-{end-1} (共 {end-start} 个)")
        actor = RemoteExpertNode.options(
            resources={f"worker_{worker_id}": 1},
            num_gpus=1
        ).remote(config, split_path, expert_start=start, expert_end=end)
        expert_ranges_with_actors.append((start, end, actor))
    print(f"✅ {NUM_EXPERT_WORKERS} 个远程专家节点已创建")

    # ---- 构建全局路由表: expert_actor_map[expert_id] -> actor ----
    expert_actor_map = [None] * 60
    for start, end, actor in expert_ranges_with_actors:
        for eid in range(start, end):
            expert_actor_map[eid] = actor
    print(f"✅ 专家路由表已构建: {sum(1 for a in expert_actor_map if a is not None)}/60 专家已分配")

    # 加载主控模型
    num_gpus = torch.cuda.device_count()
    master_device = "cuda:1" if num_gpus > 1 else "cuda:0"
    print(f"✅ 检测到 {num_gpus} 张GPU，主节点使用 {master_device}")

    model = Qwen2MoeForCausalLM(config).to(master_device).bfloat16()
    base_sd = torch.load(os.path.join(split_path, "base_model.pt"), map_location=master_device)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    if missing:
        print(f"⚠️  load_state_dict 缺失的 key ({len(missing)} 个):")
        for k in missing[:5]:
            print(f"   - {k}")
        if len(missing) > 5:
            print(f"   ... 及其他 {len(missing) - 5} 个")
    if unexpected:
        print(f"⚠️  load_state_dict 多余的 key ({len(unexpected)} 个):")
        for k in unexpected[:5]:
            print(f"   - {k}")
        if len(unexpected) > 5:
            print(f"   ... 及其他 {len(unexpected) - 5} 个")
    print("✅ 主控模型加载完成")

    # 关联路由表 + 层号到每个 MoE 层
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            layer.mlp.expert_actor_map = expert_actor_map
            layer.mlp.layer_idx = idx

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("⏳ 等待远程专家节点完成初始化...")
    time.sleep(10)

    print("="*60)
    print("✅ 分布式MoE推理服务初始化完成！")
    print("="*60 + "\n")

# ==========================================
# API 路由
# ==========================================

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    initialize_model()

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    print("\n🛑 正在关闭服务，清理资源...")
    ray.shutdown()
    print("✅ 资源清理完成")

@app.get("/", tags=["系统"])
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "分布式MoE推理服务",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy" if model is not None else "initializing",
        model_loaded=model is not None,
        gpu_count=torch.cuda.device_count(),
        device=str(master_device) if master_device else "unknown"
    )

@app.post("/inference", response_model=InferenceResponse, tags=["推理"])
async def inference(request: InferenceRequest):
    """
    执行推理请求
    
    - **prompt**: 输入的问题或提示文本
    - **max_new_tokens**: 生成的最大token数（1-200）
    - **temperature**: 温度参数，控制随机性（0.1-2.0）
    - **top_p**: nucleus sampling参数（0.0-1.0）
    - **do_sample**: 是否使用采样
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型尚未初始化完成，请稍后重试")
    
    try:
        # 编码输入
        inputs = tokenizer(request.prompt, return_tensors="pt").to(master_device)
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        elapsed_time = time.time() - start_time
        
        # 解码输出
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return InferenceResponse(
            success=True,
            result=result,
            prompt=request.prompt,
            elapsed_time=round(elapsed_time, 2),
            num_tokens=num_tokens,
            error=None
        )
        
    except Exception as e:
        return InferenceResponse(
            success=False,
            result="",
            prompt=request.prompt,
            elapsed_time=0,
            num_tokens=0,
            error=str(e)
        )

# ==========================================
# 启动服务
# ==========================================
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,        # 端口号
        log_level="info"
    )
