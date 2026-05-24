# moe-dis 代码变更记录

本文档汇总从代码审计到部署调试过程中，对 moe-dis 项目所有文件的修改。

---

## 1. 共享专家 (shared_expert) — 核心设计

**影响文件**: `master/split_weights.py`, `master/master_node_api.py`

**设计决策**: 共享专家不分布到 Worker，始终在 Master 本地 GPU 计算。

**实现**:
- `split_weights.py` 第 40 行: `if ".mlp.experts." in key:` 作为判断条件，`shared_expert` 的 key 格式为 `model.layers.X.mlp.shared_expert.*`，不包含 `.mlp.experts.`，其权重天然留在 `base_model.pt` 中，不会被拆分到 `expert_N.pt`。
- `lean_moe_init` (第 88-97 行): monkey-patch 中 `shared_expert` 和 `shared_expert_gate` 在 Master 本地创建，不参与远程调度。
- `distributed_forward_patch` (第 132-133 行): 共享专家在 Master 本地前向计算，输出与远程 expert 结果相加。

**依据**: Qwen1.5-MoE 的 shared_expert 有 `intermediate_size=5632`，参数量大，且所有 token 都会经过它。如果分布到 Worker 上，每次前向都需要传输 hidden_states + 结果，通信开销大。保留本地避免了网络瓶颈。

---

## 2. distributed_forward_patch — 推理路由修复

**影响文件**: `master/master_node_api.py`

### 2.1 norm_topk_prob 条件守卫 (Bug 1, Critical)

Qwen1.5-MoE-A2.7B 的 config 中 `norm_topk_prob=False`，但 patch 代码无条件执行了 `routing_weights /= routing_weights.sum(dim=-1, keepdim=True)`，导致路由权重被错误归一化。

```python
# 修复前 (无条件归一化)
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

# 修复后 (按 config 判断)
if self.norm_topk_prob:
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
```

### 2.2 routing_weights dtype 转换 (Bug 1, Critical)

新增 `routing_weights.to(hidden_states.dtype)` 将 float32 的路由权重转为 bfloat16，确保与 hidden_states 的 dtype 一致，避免隐式类型转换的精度损失。

```python
routing_weights = routing_weights.to(hidden_states.dtype)
```

### 2.3 num_experts 硬编码修复

```python
# 修复前
expert_mask = F.one_hot(selected_experts, num_classes=60)

# 修复后
expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
```

### 2.4 expert_hit 优化

```python
# 修复前: 遍历全部 60 个 expert
for expert_idx in range(60):

# 修复后: 只遍历当前 batch 实际被选中的 expert
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
for expert_idx in expert_hit:
```

---

## 3. GPU/CPU 分层加载 — 索引修复 (Bug 2)

**影响文件**: `master/worker_node.py`, `worker_0/worker_node.py`, `worker_1/worker_node.py`

`load_to_gpu` 判断使用全局 expert ID 而非本节点相对索引，导致 Expert 范围非零起点的 Worker 将所有专家加载到 CPU。

```python
# 修复前 (eid 是全局 ID，Worker 1 的 eid 从 30 开始，永远 > gpu_capacity)
load_to_gpu = (eid < self.gpu_capacity)

# 修复后 (使用相对索引)
load_to_gpu = ((eid - expert_start) < self.gpu_capacity)
```

例如 Worker 1 负责 expert 30-59，gpu_capacity=29。原代码 `30 < 29` 为 False，60 个专家全部入 CPU。

---

## 4. Worker import 路径修复

**影响文件**: `master/worker_node.py`, `worker_0/worker_node.py`, `worker_1/worker_node.py`

```python
# 修复前: 依赖当前工作目录
sys.path.append(os.path.abspath("./qwen_src"))

# 修复后: 基于文件自身位置
_worker_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_worker_dir, "qwen_src")):
    sys.path.insert(0, _worker_dir)
```

---

## 5. Ray 资源调度修复

**影响文件**: `master/master_node_api.py`, `README.md`

```bash
# 修复前: worker_id: 0 的资源量为 0，Ray 调度时任何节点都满足
--resources='{"worker_id": 0}'

# 修复后: 每个 Worker 用独立的资源名
--resources='{"worker_0": 1}'
--resources='{"worker_1": 1}'
```

```python
# master_node_api.py actor 创建
# 修复前
resources={"worker_id": worker_id}

# 修复后
resources={f"worker_{worker_id}": 1}
```

**后果**: 修复前 Worker 0 的 actor 被调度到 Master 的 GPU 上，与 base model 抢显存导致 OOM。

---

## 6. GPU 显存容量调整

**影响文件**: `master/worker_node.py`, `worker_0/worker_node.py`, `worker_1/worker_node.py`

```python
# 修复前: 75% 用于加载专家模型，25% 预留推理
available_vram_gb = total_vram_gb * 0.75

# 修复后: 65% 用于加载专家模型，35% 预留推理
available_vram_gb = total_vram_gb * 0.65
```

**原因**: 实际显存开销（PyTorch 框架开销、CUDA context、ModuleDict 结构）比估算值大，75% 下显存占用达 87%，推理时 OOM 风险高。

---

## 7. CUDA 环境变量

**影响文件**: `master/worker_node.py`, `worker_0/worker_node.py`, `worker_1/worker_node.py`

```python
# 修复前
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 修复后 (新增 max_split_size_mb + CUDA_ALLOC_CONF)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

限制显存分配碎片大小，减少大块连续分配失败的概率。

---

## 8. 多 Worker 扩展

**影响文件**: `master/master_node_api.py`

新增 `NUM_EXPERT_WORKERS` 变量和自动分区逻辑（第 142-151 行）：

```python
NUM_EXPERT_WORKERS = 2  # 可改为 3 扩展更多 Worker
_experts_per_worker = 60 // NUM_EXPERT_WORKERS
EXPERT_RANGES = []
for i in range(NUM_EXPERT_WORKERS):
    s = i * _experts_per_worker
    e = s + _experts_per_worker if i < NUM_EXPERT_WORKERS - 1 else 60
    EXPERT_RANGES.append((s, e))
```

创建 `worker_1/` 目录用于第三台机器。

---

## 9. 诊断日志增强

**影响文件**: `master/master_node_api.py`, `master/worker_node.py`

- `master_node_api.py`: `load_state_dict` 后打印 missing/unexpected keys，便于验证权重完整性
- `master/worker_node.py`: 加载完成后三重 `gc.collect()`，最终显示 VRAM 和 RAM 占用

---

## 修改清单

| 文件 | 修改内容 |
|------|---------|
| `master/master_node_api.py` | 推理路由修复 (norm_topk_prob, dtype, num_experts, expert_hit); 多 Worker 扩展; 资源调度修复; 诊断日志 |
| `master/worker_node.py` | GPU/CPU 索引修复; import 路径修复; CUDA 环境变量; GPU 容量 75%→65%; 内存清理优化 |
| `worker_0/worker_node.py` | 同上 (master/worker_node.py 的所有修改) |
| `worker_1/worker_node.py` | 同上; 新创建目录 |
| `master/split_weights.py` | 无修改 (共享专家逻辑原本正确) |
| `README.md` | 部署文档; 资源命名修复; 验证命令更新 |
