"""
Minimal test: 直接调 DashScope embedding API 看底层的真实错误信息
"""
import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-d0243c96bc094ea8b912cd052c509bbf'

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Test 1: 单条简单文本
print("Test 1: single simple text...")
try:
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=["hello world"],
    )
    print(f"  OK! dim={len(resp.data[0].embedding)}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: 批量 5 条
print("\nTest 2: batch of 5...")
try:
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=["hello", "world", "test", "foo", "bar"],
    )
    print(f"  OK! n={len(resp.data)}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: 长文本（模拟论文内容）
print("\nTest 3: long text (~2000 chars)...")
long_text = "This is a test sentence about deep learning and neural networks. " * 100
try:
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=[long_text],
    )
    print(f"  OK! dim={len(resp.data[0].embedding)}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: text-embedding-v1 (原始模型)
print("\nTest 4: text-embedding-v1 original model...")
try:
    resp = client.embeddings.create(
        model="text-embedding-v1",
        input=["hello world test"],
    )
    print(f"  OK! dim={len(resp.data[0].embedding)}")
except Exception as e:
    print(f"  FAIL: {e}")
