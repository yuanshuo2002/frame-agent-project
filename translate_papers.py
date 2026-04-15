#!/usr/bin/env python3
"""提取 FRAME (Ours) 的 2 篇论文，调用 vLLM 翻译为完整中文 Markdown"""

import json
import sys
import os
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import requests

# === 配置 ===
INPUT = "results/inference/comparison_results.json"
OUTPUT = "FRAME_Generated_Papers_CN.md"
PAPER_INDICES = [0, 2]  # 第1篇(CT影像) 和 第3篇(联邦学习)
API_URL = "http://localhost:10088/v1/chat/completions"
MODEL_ID = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

SECTION_INFO = {
    "topic": ("一、引言与研究背景", "阐述研究主题的背景、问题定义、研究意义与创新点"),
    "background": ("二、领域背景", "介绍相关领域的技术发展历程、核心概念与理论基础"),
    "related_work": ("三、相关工作", "综述已有研究方法，对比分析与本文方法的异同"),
    "methodology": ("四、方法论", "详细描述研究采用的数据集、模型架构、实验设计与训练策略"),
    "result": ("五、实验结果", "展示定量与定性实验结果、性能指标对比及消融实验"),
    "conclusion": ("六、结论与展望", "总结主要贡献、分析局限性并指出未来研究方向"),
}

def call_vllm(prompt: str, max_tokens: int = 2048) -> str:
    """调用本地 vLLM 进行翻译"""
    try:
        resp = requests.post(
            API_URL,
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        result = resp.json()

        # 检查错误
        if "error" in result:
            print(f"    [WARN] API error: {result['error']}", flush=True)
            return f"[翻译服务暂时不可用]"

        if "choices" not in result or len(result["choices"]) == 0:
            print(f"    [WARN] No choices in response: {str(result)[:300]}", flush=True)
            time.sleep(3)
            # 重试一次
            resp2 = requests.post(API_URL, json={
                "model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3, "max_tokens": max_tokens,
            }, timeout=180)
            result2 = resp2.json()
            if "choices" not in result2:
                return "[翻译失败，原文如下]\n\n" + prompt[prompt.index("原文："):].replace("---", "").strip() if "原文：" in prompt else "[翻译失败]"
            result = result2

        text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    [ERROR] vLLM call failed: {e}", flush=True)
        return "[翻译失败，原文如下]\n\n" + (prompt[prompt.index("原文："):].replace("---", "").strip() if "原文：" in prompt else "[翻译失败]")

    # 清理可能的 markdown code block 包裹
    if text.startswith("```") and "```" in text[3:]:
        lines = text.split("\n")
        new_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if line.strip():
                new_lines.append(line)
        text = "\n".join(new_lines).strip()

    return text


def translate_section(text: str, section_name: str) -> str:
    """将英文章节翻译为中文，支持长文本分段"""
    if not text or len(text.strip()) < 10:
        return text or ""

    MAX_CHUNK = 1200  # 每段最大字符数（确保 prompt+output < 4096 tokens）
    words = text.split()

    if len(words) <= 200:  # 短文本直接翻译
        prompt = f"""你是一位专业的学术翻译。将以下英文学术文本翻译成中文。

要求：
- 保持学术写作风格，术语准确
- 保留段落结构和格式（加粗、列表、表格等）
- 保留引用标记如 [1], [2] 等
- 不要添加任何解释或注释，只输出中文翻译

章节：{section_name}

原文：
---
{text}
---

中文翻译："""
        return call_vllm(prompt)

    # 长文本分段翻译
    chunks = []
    current_chunk = []
    current_len = 0

    # 先按段落切分
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > MAX_CHUNK and current_chunk:
            chunks.append(" ".join(current_chunk) if isinstance(current_chunk[0], str) else "\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # 逐段翻译
    translated_parts = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        print(f"    chunk {idx+1}/{total} ({len(chunk)} chars)...", flush=True)
        prompt = f"""你是一位专业的学术翻译。将以下英文学术文本片段翻译成中文。

要求：
- 保持学术写作风格，术语准确
- 保留段落结构、加粗、列表等格式
- 保留引用标记 [1], [2] 等
- 这是第 {idx+1}/{total} 段，只翻译这一段
- 不要添加任何解释，直接输出中文翻译

章节：{section_name}

原文片段：
---
{chunk}
---

中文翻译："""

        translated = call_vllm(prompt)
        translated_parts.append(translated)
        time.sleep(0.5)  # 节流

    return "\n\n".join(translated_parts)


def main():
    print(f"\n{'='*60}")
    print("FRAME (Ours) -> 中文翻译 | vLLM Qwen2.5-7B")
    print(f"{'='*60}")

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    ours = data.get("ours", [])
    selected = [ours[i] for i in PAPER_INDICES if i < len(ours)]
    print(f"\n  选择 {len(selected)} 篇论文 (索引 {PAPER_INDICES})")

    md_lines = []
    md_lines.append("# FRAME (Ours) 生成论文选集（中文版）\n")
    md_lines.append("> 本文档包含由 FRAME 完整流水线生成的 **2 篇**医学研究论文的中文全译。\n")
    md_lines.append("")
    md_lines.append("| 属性 | 说明 |")
    md_lines.append("|:-----|:-----|")
    md_lines.append("| **生成框架** | FRAME (Feedback-Refined Agent Methodology) |")
    md_lines.append("| **生成流程** | RAG 检索 → Filter 过滤 → Integrator 整合 → Generator 生成 |")
    md_lines.append("| **基础模型** | Qwen2.5-7B-Instruct (vLLM, RTX 3090) |")
    md_lines.append("| **Embedding** | text-embedding-v3 (阿里云百炼) |")
    md_lines.append("| **训练数据** | 20 篇 × 6 章 = 120 样本 (G-E-R 双轮迭代) |")
    md_lines.append("| **翻译模型** | Qwen2.5-7B-Instruct (同上, vLLM 本地) |")
    md_lines.append("")
    md_lines.append("---\n")

    for pi, paper in enumerate(selected):
        title_en = paper.get("research_topic", f"Paper {pi+1}")
        timestamp = paper.get("timestamp", "")
        secs = paper.get("sections", {})

        total_chars = sum(s.get("content_length", 0) for s in secs.values())

        # 论文标题（翻译）
        print(f"\n{'─'*50}")
        print(f"[{pi+1}] Translating TITLE: {title_en[:60]}...")
        title_cn = call_vllm(
            f'将以下学术论文标题翻译成简洁的中文标题。只输出中文翻译，不要解释：\n"{title_en}"',
            max_tokens=256,
        )

        domain = ["医学影像与深度学习", "隐私保护与联邦学习"][pi]

        md_lines.append(f"\n## 论文 {pi + 1}：{title_cn}\n")
        md_lines.append(f"> *原标题：{title_en}*\n")
        md_lines.append(f"| 属性 | 值 |")
        md_lines.append(f"|:-----|:---|")
        md_lines.append(f"| **所属领域** | {domain} |")
        md_lines.append(f"| **生成方法** | Ours (完整 FRAME 流水线) |")
        md_lines.append(f"| **原文总字符数** | {total_chars:,} |")
        if timestamp:
            md_lines.append(f"| **生成时间** | {timestamp} |")
        md_lines.append("")

        # 各章节
        for sk in ["topic", "background", "related_work", "methodology", "result", "conclusion"]:
            cn_name, desc = SECTION_INFO[sk]
            sec_data = secs.get(sk, {})
            content = sec_data.get("generated_content", "")

            if not content:
                continue

            clen = len(content)
            print(f"\n  [{cn_name}] ({clen} chars) translating...", flush=True)
            t0 = time.time()

            translated = translate_section(content, cn_name)
            elapsed = time.time() - t0
            out_len = len(translated)

            print(f"  [{cn_name}] DONE in {elapsed:.1f}s | {out_len} chars", flush=True)

            md_lines.append(f"\n### {cn_name}\n")
            md_lines.append(f"> {desc}\n")
            md_lines.append(translated)
            md_lines.append("")  # 段后空行

        md_lines.append("\n---\n")

    # 写入文件
    out_text = "\n".join(md_lines)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(out_text)

    total_out = len(out_text)
    print(f"\n{'='*60}")
    print(f"DONE: {OUTPUT}")
    print(f"   Papers: {len(selected)}, Output: {total_out:,} chars")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
