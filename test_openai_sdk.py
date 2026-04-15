"""用 OpenAI SDK 精确模拟 FRAME 的调用方式，定位 502 根因"""
import json
import time
from openai import OpenAI

# 和 FRAME 完全一样的初始化方式
client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:10088/v1",
    timeout=60.0,
    max_retries=0,
)

print("="*60)
print("测试: OpenAI SDK 调用 (和 FRAME 一模一样)")
print("="*60)

# 模拟 Generator 的实际调用
messages = [
    {"role": "system", "content": "You are an expert academic writing assistant."},
    {"role": "user", "content": """Write a topic/introduction section for this paper:

Research Topic: ImageJ2: ImageJ for the next generation of scientific image data.

Reference Materials:
This paper presents ImageJ2, a major redesign of the popular ImageJ image processing software for scientific imaging. 
The new architecture addresses key limitations of the original ImageJ including limited data type support, 
lack of modern interoperability standards, and constrained extensibility mechanisms. 

ImageJ2 introduces several fundamental improvements:
1. A new data model based on N-dimensional pixel arrays with support for arbitrary axis types
2. Integration with SciJava plugin framework enabling dynamic discovery and lifecycle management
3. Enhanced support for large datasets through lazy loading and chunked access patterns

Requirements:
- Provide comprehensive background on the research problem
- Explain the significance and motivation
- Write in formal academic English"""}
]

print(f"发送请求...")
print(f"  messages: {len(messages)} 条")
print(f"  总字符数: {sum(len(m['content']) for m in messages)}")
print(f"  model: Qwen/Qwen2.5-7B-Instruct")
print(f"  temperature: 0.7")
print(f"  max_tokens: 2048")
print()

try:
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f"✅ 成功! 耗时={elapsed:.1f}s")
    print(f"  响应长度: {len(content)} 字符")
    print(f"  前200字符: {content[:200]}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")

# 测试 2: 带 stream=False 显式指定（某些 vLLM 版本默认 stream）
print("\n" + "="*60)
print("测试 2: OpenAI SDK + stream=False 显式指定")
print("="*60)
try:
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50,
        stream=False,
    )
    elapsed = time.time() - start
    print(f"✅ 成功! 耗时={elapsed:.1f}s")
    print(f"  响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")

# 测试 3: 用 chat_json 方式（带 response_format）
print("\n" + "="*60)
print("测试 3: OpenAI SDK + response_format=json_object (Evaluator模式)")
print("="*60)
try:
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are an expert evaluator. Return JSON only."},
            {"role": "user", "content": 'Evaluate: "This paper presents ImageJ2, a major redesign of ImageJ software." Return JSON with score and summary.'}
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
        stream=False,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f"✅ 成功! 耗时={elapsed:.1f}s")
    print(f"  响应: {content[:300]}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")
