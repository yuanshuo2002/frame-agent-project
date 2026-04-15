"""独立对比实验脚本 - 绕过日志系统直接输出到 stdout"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 60)
print("FRAME Phase 4: Comparison Experiment")
print("=" * 60)

# Step 1: Build FAISS index
print("\n[Step 1] Building FAISS index...", flush=True)
from src.inference.pipeline import load_or_build_faiss_index, FrameInferencePipeline

faiss_store = load_or_build_faiss_index()
print(f"  FAISS OK: vectors={faiss_store.total_vectors}, dim={faiss_store.dimension}", flush=True)

# Step 2: Create pipeline
print("\n[Step 2] Creating pipeline...", flush=True)
pipeline = FrameInferencePipeline(faiss_store=faiss_store)

# Step 3: Define topics
topics = [
    "Deep Learning Approaches for CT-based Medical Image Analysis",
    "NLP Methods for Clinical Note Information Extraction",
    "Federated Learning Frameworks for Privacy-Preserving Medical AI",
    "Automated Radiology Report Generation from Medical Images",
    "Graph Neural Networks for Molecular Property Prediction",
]
methods = ["no_rag", "rag", "ours"]

# Step 4: Run comparison
import time
import json
results = {m: [] for m in methods}

total = len(topics) * len(methods)
done = 0

print(f"\n[Step 3] Running {total} generations ({len(topics)} topics x {len(methods)} methods)...", flush=True)

for ti, topic in enumerate(topics):
    print(f"\n{'─'*50}", flush=True)
    print(f"[{ti+1}/{len(topics)}] {topic[:55]}...", flush=True)
    
    for method in methods:
        done += 1
        t0 = time.time()
        try:
            paper = pipeline.generate_full_paper(topic, method=method)
            # Check if content was actually generated
            has_content = False
            total_chars = 0
            for sk, sv in paper.get("sections", {}).items():
                c = sv.get("generated_content", "")
                total_chars += len(c)
                if len(c) > 100 and "Certainly!" not in c:
                    has_content = True
            
            results[method].append(paper)
            elapsed = time.time() - t0
            status = "OK" if has_content else f"EMPTY({total_chars}chars)"
            print(f"  [{done}/{total}] {method:>8s}: {status} ({elapsed:.1f}s, {total_chars} chars)", flush=True)
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{done}/{total}] {method:>8s}: ERROR - {e} ({elapsed:.1f}s)", flush=True)
            import traceback
            traceback.print_exc()

# Step 5: Save
out_path = r"c:\Users\11380\WorkBuddy\20260410152734\frame_reproduction\results\inference\comparison_results.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Step 6: Summary
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print("=" * 60)

for m in methods:
    papers = results[m]
    ok = sum(1 for p in papers if p.get("sections"))
    total_c = sum(
        len(s.get("generated_content", ""))
        for p in papers
        for s in p.get("sections", {}).values()
    )
    print(f"  {m:>8s}: {ok}/{len(papers)} papers | {total_c:,} chars")

print(f"\nSaved to: {out_path}")
print(f"File size: {os.path.getsize(out_path):,} bytes")
print("\nDONE!")
