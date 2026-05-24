import sys
import os
import torch
from transformers import AutoTokenizer

# 1. 强制将本地 qwen_src 加入路径，确保 import 不会去 site-packages 找
sys.path.append(os.path.join(os.getcwd(), "qwen_src"))

from qwen_src.configuration_qwen2_moe import Qwen2MoeConfig
from qwen_src.modeling_qwen2_moe import Qwen2MoeForCausalLM

model_path = "./weights/qwen1.5-moe-2.7b-chat"

# 2. 加载配置和模型
print("正在从本地加载模型...")
config = Qwen2MoeConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 使用 bfloat16 节省显存，并自动映射到可用设备
model = Qwen2MoeForCausalLM.from_pretrained(
    model_path,
    config=config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 3. 测试推理
prompt = "什么是人工智能？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("开始生成...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))