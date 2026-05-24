# MoE 分布式推理部署文档

Qwen1.5-MoE-A2.7B 模型的多机分布式推理系统，基于 Ray 集群将 60 个专家（expert）分区到多台 GPU 机器上并行计算。

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    机器1: Master (Master 节点)                     │
│                                                                   │
│  Ray Head + FastAPI (:8000)                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  base_model.pt                                               │ │
│  │  - Embedding (embed_tokens)                                  │ │
│  │  - 24层 Attention (Q/K/V/O projection)                       │ │
│  │  - 24层 Router Gate (gate Linear)                            │ │
│  │  - 24层 Shared Expert (shared_expert + shared_expert_gate)   │ │
│  │  - LayerNorm, lm_head                                        │ │
│  │                                                              │ │
│  │  推理流程:                                                    │ │
│  │  1. Tokenize → Embedding                                     │ │
│  │  2. For each layer:                                          │ │
│  │     a. Self-Attention (本地 GPU)                              │ │
│  │     b. Router Gate 计算 top-4 experts (本地 GPU)              │ │
│  │     c. 按 expert_actor_map 路由到对应 Worker (Ray remote)     │ │
│  │     d. Shared Expert (本地 GPU)                               │ │
│  │  3. lm_head → Decode                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Ray cluster
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 机器2: Worker 0   │ │ 机器3: Worker 1   │ │ 机器N: Worker N-1 │
│ experts 0-29     │ │ experts 30-59    │ │ ...              │
│                  │ │                  │ │                  │
│ GPU: experts 0-K │ │ GPU: experts     │ │                  │
│ CPU: experts K-29│ │   30-30+M        │ │                  │
│                  │ │ CPU: experts     │ │                  │
│ worker_id: 0     │ │   30+M-59        │ │                  │
│                  │ │ worker_id: 1     │ │ worker_id: N-1   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 总 expert 数 | 60 | `config.num_experts` |
| top-k | 4 | 每个 token 路由到 4 个 expert |
| hidden_size | 2048 | 隐藏维度 |
| moe_intermediate_size | 1408 | 每个 routed expert 的 FFN 中间维度 |
| shared_expert_intermediate_size | 5632 | 共享 expert 的 FFN 中间维度 |
| num_hidden_layers | 24 | Transformer 层数 |
| 单个 expert 参数量 (24层) | ~207M | 约 396MB (bfloat16) |
| 30 个 expert 参数量 | ~6.2B | 约 11.6GB (bfloat16) |
| base_model 参数量 | ~1.3B | 约 2.5GB (bfloat16) |

### 数据流

```
输入 prompt
  │
  ▼
Tokenizer ──► Embedding
  │
  ▼
For layer 0..23:
  ├─ Self-Attention (本地 GPU)
  ├─ Router Gate ──► softmax ──► top-4 experts
  │                                  │
  │    ┌─────────────────────────────┘
  │    ▼
  │   对每个被选中的 expert_idx:
  │     expert_actor_map[idx].compute_expert.remote(layer, expert_id, tokens, weights)
  │       │
  │       ├─► Worker 0: L{layer}_E{0..29}  ──► Qwen2MoeMLP forward
  │       └─► Worker 1: L{layer}_E{30..59} ──► Qwen2MoeMLP forward
  │    │
  │    ▼
  │   ray.get(futures) ──► index_add_ 聚合
  ├─ Shared Expert (本地 GPU)
  └─ Residual Add
  │
  ▼
LayerNorm ──► lm_head ──► Decode ──► 输出文本
```

## 目录结构

```
moe-dis/
├── README.md                        # 本文档
│
├── master/                          # === 机器1 (Master) 部署代码 ===
│   ├── master_node_api.py           # FastAPI 推理服务 + 多 worker 路由引擎
│   ├── worker_node.py               # RemoteExpertNode Ray actor 类定义
│   ├── split_weights.py             # 权重切分脚本 (safetensors → base_model + experts)
│   ├── print_model_struct.py        # 模型结构诊断工具
│   ├── test.py                      # 本地单机测试脚本
│   ├── client_example.py            # HTTP API 调用示例
│   ├── qwen_src/                    # Qwen2MoE 模型源码
│   │   ├── __init__.py
│   │   ├── configuration_qwen2_moe.py
│   │   └── modeling_qwen2_moe.py
│   ├── weights/                     # 原始模型文件 (tokenizer + config)
│   │   └── qwen1.5-moe-2.7b-chat/
│   └── split_weights_chat/          # 切分后的权重
│       ├── base_model.pt            # 基础模型权重 (~2.5GB)
│       └── experts/                 # 60个expert权重文件
│           ├── expert_0.pt          # (~396MB each)
│           ├── expert_1.pt
│           └── ...
│
├── worker_0/                        # === 机器2 (Worker 0) 部署代码 ===
│   ├── worker_node.py               # RemoteExpertNode Ray actor (同 master 版)
│   ├── split_weights.py             # 权重切分脚本
│   ├── qwen_src/                    # Qwen2MoE 模型源码 (同 master 版)
│   └── split_weights_chat/experts/  # 需要全部 60 个 expert_N.pt
│
└── worker_1/                        # === 机器3 (Worker 1) 部署代码 ===
    ├── worker_node.py               # RemoteExpertNode Ray actor (同 worker_0)
    ├── split_weights.py             # 权重切分脚本
    ├── qwen_src/                    # Qwen2MoE 模型源码
    └── split_weights_chat/experts/  # 需要全部 60 个 expert_N.pt
```

### 文件功能速查

| 文件 | 在哪里运行 | 作用 |
|------|-----------|------|
| `master/master_node_api.py` | Master 机器 | 启动 FastAPI 推理服务，创建 Ray actors，路由 expert 计算 |
| `master/worker_node.py` | Master 机器 (类定义) | `RemoteExpertNode` Ray actor 的类定义，Master 通过此文件创建远程 actor |
| `worker_0/worker_node.py` | Worker 0 (被 Ray 调度) | Worker 上实际运行的代码，Ray 将 actor 调度到此处 |
| `worker_1/worker_node.py` | Worker 1 (被 Ray 调度) | 同上，负责另一段 expert 范围 |
| `**/split_weights.py` | 任一台机器 | 将 HuggingFace safetensors 按 expert 拆分为 1 个 base_model.pt + 60 个 expert_N.pt |
| `master/test.py` | Master 机器 | 不经过 Ray 的本地单机推理，用于验证模型权重正确性 |
| `master/client_example.py` | 任意机器 | 向 FastAPI 发送 HTTP 推理请求的示例客户端 |
| `master/print_model_struct.py` | Master 机器 | 诊断工具：打印模型结构、检查拆分后的权重文件是否正确 |

## 部署步骤

### 前置要求

- 每台机器: Ubuntu 22.04+, CUDA 12.x, Python 3.11, conda
- 每台 GPU 机器: >= 24GB VRAM 推荐 (30 个 expert 约需 12GB + 推理时激活值)
- Master 机器: >= 16GB VRAM (base model ~2.5GB + KV cache)
- 所有机器之间网络互通，防火墙放通 Ray 端口 (默认 6379, 8265, 10001-10999)

### 步骤 1: 环境准备 (所有机器)

```bash
# 创建 conda 环境
conda create -n moe python=3.11 -y
conda activate moe

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ray transformers accelerate bitsandbytes safetensors psutil tqdm huggingface-hub fastapi uvicorn pydantic

# 验证
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 步骤 2: 下载模型 + 权重切分 (在 Master 机器上执行一次)

```bash
cd /home/oai/moe-dis/master

# 下载 Qwen1.5-MoE-A2.7B-Chat
huggingface-cli download --resume-download Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --local-dir ./weights/qwen1.5-moe-2.7b-chat \
    --local-dir-use-symlinks False

# 切分权重
python split_weights.py
# 输出: split_weights_chat/base_model.pt + split_weights_chat/experts/expert_0.pt ... expert_59.pt
```

### 步骤 3: 权重分发 (从 Master → Worker 机器)

```bash
# 方式1: rsync (推荐)
rsync -avz --progress /home/oai/moe-dis/master/split_weights_chat/experts/ \
    worker0:/home/oai/moe-dis/split_weights_chat/experts/

rsync -avz --progress /home/oai/moe-dis/master/split_weights_chat/experts/ \
    worker1:/home/oai/moe-dis/split_weights_chat/experts/

# 方式2: scp
scp -r /home/oai/moe-dis/master/split_weights_chat/experts/ \
    worker0:/home/oai/moe-dis/split_weights_chat/experts/
```

同时分发源码:

```bash
# Worker 0
rsync -avz /home/oai/moe-dis/worker_0/ worker0:/home/oai/moe-dis/

# Worker 1
rsync -avz /home/oai/moe-dis/worker_1/ worker1:/home/oai/moe-dis/
```

### 步骤 4: 启动 Ray 集群

**机器1 (Master, IP 以实际为准):**

```bash
cd /home/oai/moe-dis
ray stop --force
ray start --head --temp-dir=/home/oai/ray_tmp --dashboard-host=0.0.0.0
# Ray Dashboard: http://10.29.155.44:8265
```

**机器2 (Worker 0):**

```bash
ray stop --force
ray start --address='10.29.155.44:6379' --resources='{"worker_id": 0}' --num-gpus=1
```

**机器3 (Worker 1):**

```bash
ray stop --force
ray start --address='10.29.155.44:6379' --resources='{"worker_id": 1}' --num-gpus=1
```

### 步骤 5: 验证集群状态

在 Master 机器上:

```bash
# 查看集群节点
ray status

# 预期输出:
# 3 nodes
#  node1 (head): 1 GPU
#  node2 (worker): 1 GPU, resources: {"worker_id": 0}
#  node3 (worker): 1 GPU, resources: {"worker_id": 1}

# 查看 Dashboard
# 浏览器打开: http://10.29.155.44:8265 → Nodes 页签
```

### 步骤 6: 启动推理服务

```bash
cd /home/oai/moe-dis/master
python master_node_api.py
```

预期启动日志:

```
============================================================
🚀 正在初始化分布式MoE推理服务（多机版）...
   专家分区: [(0, 30), (30, 60)]
============================================================
✅ 已连接到Ray集群
  创建 Worker 0: 负责专家 0-29 (共 30 个)
  创建 Worker 1: 负责专家 30-59 (共 30 个)
✅ 2 个远程专家节点已创建
✅ 专家路由表已构建: 60/60 专家已分配
✅ 检测到 2 张GPU，主节点使用 cuda:1
✅ 主控模型加载完成
⏳ 等待远程专家节点完成初始化...
✅ 分布式MoE推理服务初始化完成！
```

### 步骤 7: 测试推理

```bash
# 方式1: curl
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "什么是人工智能？", "max_new_tokens": 50}'

# 方式2: Python 客户端
cd /home/oai/moe-dis/master
python client_example.py

# 方式3: Swagger UI
# 浏览器打开: http://10.29.155.44:8000/docs
```

## 验证清单

### 权重切分验证

```bash
cd /home/oai/moe-dis/master
python print_model_struct.py

# 检查:
# 1. base_model.pt 不包含 expert 参数 (残留数应为 0)
# 2. 每个 expert_N.pt 跨越 24 层 Transformer
# 3. base_model.pt 包含 shared_expert 参数 (24层 × 96项 = 共享专家参数)
# 4. 有效 expert 文件总数应为 60
```

### 本地单机验证 (不经过 Ray)

```bash
cd /home/oai/moe-dis/master

# 修改 test.py 中的 model_path 为实际路径后:
python test.py
```

应能正常输出推理文本，确认模型权重完整可用。

### 集群连通性验证

```bash
# 在 Master 上运行
python -c "
import ray
ray.init(address='auto')
print(ray.nodes())
print(f'可用资源: {ray.available_resources()}')
# 应看到 worker_id: 0 和 worker_id: 1
"
```

### API 健康检查

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"gpu_count":2,"device":"cuda:1"}
```

## 扩展更多 Worker

如果需要扩展到 3 个 expert worker (4 台机器，expert 按 20/20/20 分配):

### 修改配置

在 `master/master_node_api.py` 第 138 行:

```python
NUM_EXPERT_WORKERS = 3  # 从 2 改为 3
# 自动分区: [(0, 20), (20, 40), (40, 60)]
```

### 创建 worker_2 目录

```bash
cp -r /home/oai/moe-dis/worker_1 /home/oai/moe-dis/worker_2
```

### 机器5 启动 Ray

```bash
ray stop --force
ray start --address='10.29.155.44:6379' --resources='{"worker_id": 2}' --num-gpus=1
```

### 不均匀分区 (手动配置)

如果 GPU 显存不同，可以手动指定每个 worker 的 expert 范围:

```python
# 替换自动分区代码:
EXPERT_RANGES = [
    (0, 15),    # Worker 0: 15个expert (8GB卡)
    (15, 45),   # Worker 1: 30个expert (24GB卡)
    (45, 60),   # Worker 2: 15个expert (8GB卡)
]
```

## 故障排查

### Worker 启动后立刻退出

```bash
# 检查 Ray 日志
cat /tmp/ray/session_latest/logs/raylet.err
# 常见原因: GPU 不可用、CUDA 版本不匹配、资源标签写错
```

### master_node_api.py 卡在 "等待远程专家节点"

Worker 代码可能 OOM 或加载失败。检查 Worker 机器上的 Ray 日志:

```bash
cat /tmp/ray/session_latest/logs/worker-*.err
```

### compute_expert 返回全零

说明某个 expert 未被任何 worker 覆盖。检查:
1. `EXPERT_RANGES` 是否覆盖 0-59 全部范围
2. 所有 worker 是否都成功加入 Ray 集群
3. `expert_actor_map` 统计是否正确 (应为 60/60)

### Ray actor 没有调度到指定 Worker

```bash
# 确认 Worker 机器的 resources 标签与 master 代码匹配:
ray status
# 查看每个节点的 resources 字段
```

### 显存不足 (OOM)

```bash
# 减小 GPU 上的 expert 数量: 在 worker_node.py 的 _calculate_gpu_capacity 中
# 将 0.75 调小
available_vram_gb = total_vram_gb * 0.6  # 只使用 60% 显存加载 expert
```

## API 接口文档

启动服务后访问 `http://<master_ip>:8000/docs` 查看 Swagger 文档。

### POST /inference

请求:
```json
{
  "prompt": "什么是人工智能？",
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.8,
  "do_sample": true
}
```

响应:
```json
{
  "success": true,
  "result": "人工智能（Artificial Intelligence，简称AI）是...",
  "prompt": "什么是人工智能？",
  "elapsed_time": 3.45,
  "num_tokens": 50,
  "error": null
}
```

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_count": 2,
  "device": "cuda:1"
}
```

## 关键设计决策

1. **Gate 路由在 Master 本地执行**: router gate 是一个小矩阵 (2048×60)，本地计算延迟远低于跨网络传输 hidden_states。Master 只把选定的 token 子集 + 路由权重发给对应 Worker。

2. **Worker 存储使用全局 expert ID**: key 格式为 `L{layer}_E{global_expert_id}`，避免 master 需要做 ID 转换。Worker 只负责一段连续的全局 ID 范围。

3. **GPU/CPU 分层存储**: 每个 Worker 根据自身 GPU 显存动态决定多少 expert 放 GPU。超出的放 CPU，计算时自动在对应设备上执行。

4. **shared_expert 在 Master 本地**: shared_expert 是全局共享的，不需要分布式。而且它的 intermediate_size=5632 较大，放本地避免传输。

5. **`max_concurrency=4`**: 允许同一个 Worker 同时处理 4 个 expert 计算请求（不同 layer 或不同 token 组），提高 GPU 利用率。
