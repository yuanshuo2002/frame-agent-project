"""测试 vLLM 是否能处理 FRAME 真实场景的请求"""
import urllib.request
import json
import time
import sys

BASE_URL = "http://127.0.0.1:10088/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy",
}

def test_request(name, messages, max_tokens=50, use_json_format=False):
    """发送一个测试请求"""
    body = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if use_json_format:
        body["response_format"] = {"type": "json_object"}
    
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"  prompt tokens 估算: ~{sum(len(m['content'])//2 for m in messages)}")
    print(f"  max_tokens: {max_tokens}")
    print(f"  response_format: {body.get('response_format', '无')}")
    print(f"{'='*60}")

    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE_URL, data=data, headers=HEADERS)
    
    try:
        start = time.time()
        resp = urllib.request.urlopen(req, timeout=120)
        elapsed = time.time() - start
        result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"  ✅ 成功! 耗时={elapsed:.1f}s")
        print(f"  usage: prompt={usage.get('prompt_tokens','?')}, completion={usage.get('completion_tokens','?')}, total={usage.get('total_tokens','?')}")
        print(f"  响应前200字符: {content[:200]}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False


# 测试 1: 短消息 (基线)
test_request(
    "1. 短消息 (基线)",
    [{"role": "user", "content": "hello, say hi in one sentence"}],
    max_tokens=50,
)

# 测试 2: 中等长度 prompt (模拟简单章节生成)
medium_prompt = """You are an academic writing assistant. Please write a topic/introduction section for a scientific paper.

Research Topic: ImageJ2: ImageJ for the next generation of scientific image data.

Requirements:
- Provide comprehensive background on the research problem
- Explain the significance and motivation
- Write in formal academic English
- Approximately 300-500 words"""

test_request(
    "2. 中等长度 prompt (~500 chars)",
    [
        {"role": "system", "content": "You are an expert academic writer specializing in scientific papers."},
        {"role": "user", "content": medium_prompt},
    ],
    max_tokens=1024,
)

# 测试 3: 长 prompt (模拟真实 FRAME 场景)
long_context = """
This paper presents ImageJ2, a major redesign of the popular ImageJ image processing software for scientific imaging. 
The new architecture addresses key limitations of the original ImageJ including limited data type support, 
lack of modern interoperability standards, and constrained extensibility mechanisms. 

ImageJ2 introduces several fundamental improvements:
1. A new data model based on N-dimensional pixel arrays with support for arbitrary axis types
2. Integration with SciJava plugin framework enabling dynamic discovery and lifecycle management
3. Enhanced support for large datasets through lazy loading and chunked access patterns
4. Improved compatibility with other scientific computing tools via standard formats and APIs

The methodology section describes the implementation approach using a modular design pattern that separates 
concerns into distinct layers: the core image data layer, the algorithm layer, and the user interface layer.
Each component was developed following rigorous software engineering practices including unit testing, 
integration testing, and continuous integration pipelines. The evaluation was conducted across multiple 
domains including microscopy image analysis, medical imaging workflows, and general-purpose scientific visualization.

Key findings demonstrate significant performance improvements over the previous version, with particular 
gains in memory efficiency (up to 60% reduction for large datasets), processing throughput (2-3x faster 
for common operations), and developer experience metrics (reduced plugin development time by approximately 40%).
"""

test_request(
    "3. 长prompt (~1500 chars) + max_tokens=2048",
    [
        {"role": "system", "content": "You are an expert academic writing assistant. Generate high-quality scientific paper sections based on the provided context."},
        {"role": "user", "content": f"Based on the following reference material, write a comprehensive methodology section:\n\n{long_context}\n\nPlease write a detailed methodology section covering all experimental procedures, data collection methods, and analysis approaches."}
    ],
    max_tokens=2048,
)

# 测试 4: JSON 格式输出 (Evaluator 使用这个模式)
test_request(
    "4. JSON output format (Evaluator 模式)",
    [
        {"role": "system", "content": "You are an expert evaluator. Score the given text and return structured JSON."},
        {"role": "user", "content": f"Evaluate the following text:\n\n{long_context[:500]}\n\nReturn JSON with dimension_scores, overall_score, and summary."}
    ],
    max_tokens=1024,
    use_json_format=True,
)

print("\n" + "="*60)
print("全部测试完成!")
print("="*60)
