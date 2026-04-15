"""快速分析 comparison_results.json 的完整性和质量"""
import json
import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

results_path = Path(r"c:\Users\11380\WorkBuddy\20260410152734\frame_reproduction\results\inference\comparison_results.json")

with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 70)
print("📊 对比实验结果分析")
print("=" * 70)

methods = ["no_rag", "rag", "ours"]
method_labels = {"no_rag": "No-RAG (Baseline)", "rag": "RAG", "ours": "Ours (FRAME)"}
sections = ["topic", "background", "related_work", "methodology", "result", "conclusion"]

for m in methods:
    papers = data.get(m, [])
    print(f"\n{'─' * 50}")
    print(f"📂 {method_labels[m]}: {len(papers)} 篇论文")
    print(f"{'─' * 50}")
    
    total_chars = 0
    total_sections = 0
    error_sections = 0
    
    for i, paper in enumerate(papers):
        topic = paper.get("research_topic", "?")[:55]
        secs = paper.get("sections", {})
        
        sec_info = []
        for s in sections:
            if s in secs:
                sec = secs[s]
                content = sec.get("generated_content", "")
                clen = len(content)
                has_error = "error" in str(sec).lower() or not content or "Certainly!" in content or "{research_topic}" in content.lower()
                total_chars += clen
                total_sections += 1
                
                # 质量标记
                quality = "✅" if clen > 500 and not has_error else ("⚠️" if has_error else "🔸")
                sec_info.append(f"{s}={clen}{quality}")
                if has_error:
                    error_sections += 1
        
        print(f"\n  [{i+1}] {topic}...")
        print(f"      {' | '.join(sec_info)}")
    
    print(f"\n  小计: {total_sections} 章节 | {total_chars:,} 字符 | ⚠️{error_sections} 低质量章节")

# 汇总
print("\n" + "=" * 70)
print("📋 总体汇总")
print("=" * 70)

grand_total = 0
grand_errors = 0
all_papers = 0

for m in methods:
    papers = data.get(m, [])
    all_papers += len(papers)
    for paper in papers:
        for s_key, s_val in paper.get("sections", {}).items():
            content = s_val.get("generated_content", "")
            grand_total += len(content)
            if len(content) < 500 or "Certainly!" in content or "{research_topic}" in content.lower():
                grand_errors += 1

print(f"总论文数: {all_papers}")
print(f"总字符数: {grand_total:,}")
if all_papers > 0:
    print(f"低质量/模板化章节: {grand_errors}/{all_papers*6} ({100*grand_errors/(all_papers*6):.1f}%)")
else:
    print("无数据")
