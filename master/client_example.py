"""
分布式MoE推理服务 - 客户端示例
展示如何调用API进行推理
"""

import requests
import json

# API服务地址
API_URL = "http://localhost:8000"

def check_health():
    """检查服务健康状态"""
    response = requests.get(f"{API_URL}/health")
    print("🏥 健康检查:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()

def inference(prompt, max_new_tokens=512, temperature=0.7, top_p=0.8):
    """执行推理请求"""
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True
    }
    
    print(f"\n📤 发送请求: {prompt}")
    response = requests.post(f"{API_URL}/inference", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print(f"✅ 推理成功")
            print(f"⏱️  耗时: {result['elapsed_time']}s")
            print(f"📝 结果: {result['result']}")
            print(f"🔢 生成token数: {result['num_tokens']}")
        else:
            print(f"❌ 推理失败: {result['error']}")
    else:
        print(f"❌ HTTP错误: {response.status_code}")
    
    return response.json()

def main():
    print("="*60)
    print("🚀 分布式MoE推理服务 - 客户端示例")
    print("="*60)
    
    # 1. 检查服务健康状态
    health = check_health()
    if not health.get("model_loaded"):
        print("⚠️  模型尚未加载完成，请等待...")
        return
    
    # 2. 执行推理示例
    print("\n" + "="*60)
    print("开始推理测试")
    print("="*60)
    
    # 示例1
    inference("什么是人工智能？", max_new_tokens=50)
    
    # 示例2
    inference("解释深度学习的概念", max_new_tokens=80)
    
    # 示例3：自定义参数
    inference(
        "MOE模型的优势是什么？",
        max_new_tokens=512
    )

if __name__ == "__main__":
    main()

# modelscope download --model Qwen/Qwen1.5-MoE-A2.7B-Chat --local_dir /home/oai/moe-dis/weights