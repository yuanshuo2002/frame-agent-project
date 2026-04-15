#!/usr/bin/env python3
"""从 comparison_results.json 提取 Ours (FRAME) 生成的 5 篇论文，输出为 Markdown"""
import json
from pathlib import Path

INPUT = Path(__file__).parent / "results" / "inference" / "comparison_results.json"
OUTPUT = Path(__file__).parent / "FRAME_Generated_Papers.md"

with open(INPUT, 'r', encoding='utf-8') as f:
    data = json.load(f)

ours_papers = data.get("ours", [])

lines = []
lines.append("# FRAME (Ours) 生成论文集\n")
lines.append(f"> **生成方法**: FRAME 完整流水线 (RAG → Filter → Integrator → Generator)")
lines.append(f"> **模型**: Qwen2.5-7B-Instruct via vLLM (AutoDL RTX 3090)")
lines.append(f"> **论文数量**: {len(ours_papers)} 篇")
lines.append(f"> **数据来源**: `results/inference/comparison_results.json` → `ours`\n")

for idx, paper in enumerate(ours_papers, 1):
    topic = paper.get("research_topic", f"Paper {idx}")
    ts = paper.get("timestamp", "")
    
    # 统计总字符数
    sections = paper.get("sections", {})
    total_chars = sum(s.get("content_length", len(s.get("generated_content", ""))) for s in sections.values())
    
    lines.append("---")
    lines.append(f"\n## 论文 {idx}: {topic}\n")
    lines.append(f"*生成时间: {ts} | 总字符数: {total_chars:,}*\n")
    
    # 按标准论文顺序输出章节
    sec_order = ["topic", "background", "related_work", "methodology", "result", "conclusion"]
    sec_labels = {
        "topic": "Topic / Introduction",
        "background": "Background",
        "related_work": "Related Work",
        "methodology": "Methodology",
        "result": "Results",
        "conclusion": "Conclusion",
    }
    
    for sk in sec_order:
        if sk not in sections:
            continue
        sec = sections[sk]
        content = sec.get("generated_content", "").strip()
        if not content or content.lower() == "error":
            continue
        
        chars = sec.get("content_length", len(content))
        retrieved = sec.get("retrieved_count", 0)
        filtered = sec.get("filtered_count", 0)
        
        label = sec_labels.get(sk, sk.title())
        lines.append(f"### {label}\n")
        lines.append(f"*检索: {retrieved} | 过滤: {filtered} | 字符: {chars:,}*\n")
        lines.append(content)
        lines.append("")  # 章节间空行
    
    lines.append("")  # 论文间空行

# 写入
out_text = "\n".join(lines)
OUTPUT.write_text(out_text, encoding='utf-8')

# 打印摘要
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print(f"\n{'='*60}")
print(f"DONE: {OUTPUT}")
print(f"   Papers: {len(ours_papers)}")
print(f"   Total chars: {len(out_text):,}")
for i, p in enumerate(ours_papers, 1):
    secs = p.get("sections", {})
    tc = sum(s.get("content_length", 0) for s in secs.values())
    t = p['research_topic'][:50]
    print(f"   [{i}] {t}... ({tc:,} chars)")
print(f"{'='*60}")
