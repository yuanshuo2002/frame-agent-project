"""测试不同方式解决 OpenAI SDK 与 vLLM 的兼容性问题"""
import json
import time

print("="*60)
print("方案 1: 用 httpx 原始请求（绕过 OpenAI SDK）")
print("="*60)

import httpx

resp = httpx.post(
    "http://127.0.0.1:10088/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy",
    },
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 50,
    },
    timeout=60,
)
print(f"状态码: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"✅ 成功! {data['choices'][0]['message']['content']}")
else:
    print(f"❌ 失败: {resp.text[:300]}")


print("\n" + "="*60)
print("方案 2: OpenAI SDK + 自定义 httpx 客户端（去除多余头）")
print("="*60)

from openai import OpenAI
import httpx as httpx_lib

# 自定义 httpx 客户端，只保留必要头部
_custom_http = httpx_lib.Client(
    headers={
        "Content-Type": "application/json",
        # 不加额外头，让 SDK 只发最基本的
    },
    timeout=60.0,
)

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:10088/v1",
    http_client=_custom_http,
    max_retries=0,
)

try:
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=50,
    )
    print(f"✅ 成功! {resp.choices[0].message.content}")
except Exception as e:
    print(f"❌ 失败: {e}")


print("\n" + "="*60)
print("方案 3: 检查 OpenAI SDK 版本 + httpx 版本")
print("="*60)

try:
    import openai
    import httpx
    print(f"openai version: {openai.__version__}")
    print(f"httpx version: {httpx.__version__}")
except:
    pass


print("\n" + "="*60)
print("方案 4: 用 requests 库")
print("="*60)

import requests
resp = requests.post(
    "http://127.0.0.1:10088/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy",
    },
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 50,
    },
    timeout=60,
)
print(f"状态码: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"✅ 成功! {data['choices'][0]['message']['content']}")
else:
    print(f"❌ 失败: {resp.text[:300]}")
