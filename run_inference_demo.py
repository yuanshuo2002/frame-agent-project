"""
Phase 4 推理最终版 v6
完全自包含：手动建索引 + 手动调用各组件 + 统一维度=1024
"""
import os
import sys
import json
import time

os.environ['DASHSCOPE_API_KEY'] = 'sk-d0243c96bc094ea8b912cd052c509bbf'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import faiss
import requests as req_lib
from openai import OpenAI

print("=" * 60)
print("  FRAME Phase 4: Inference v6 (Final)")
print("=" * 60)

# === CONFIG (single source of truth) ===
EMB_MODEL_NAME = "text-embedding-v3"
EMB_DIM = 1024
VLLM_URL = "http://localhost:10088/v1/chat/completions"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Step 1: vLLM check
print("\n[1] Checking vLLM...")
r = req_lib.get("http://localhost:10088/v1/models", timeout=5, headers={"Authorization": "Bearer dummy"})
assert r.status_code == 200
print(f"  [OK] {json.loads(r.text)['data'][0]['id']}")

# Step 2: Embedding client
print(f"\n[2] Embedding client ({EMB_MODEL_NAME}, dim={EMB_DIM})...")
emb_client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
test_emb = emb_client.embeddings.create(model=EMB_MODEL_NAME, input=["test"])
assert len(test_emb.data[0].embedding) == EMB_DIM
print(f"  [OK] dim={len(test_emb.data[0].embedding)}")

# Step 3: Load training data & build FAISS
print("\n[3] Building FAISS index from training results...")
with open(os.path.join(PROJECT_ROOT, "checkpoints", "training_results.json"), 'r', encoding='utf-8') as f:
    train_data = json.load(f)

texts, metas = [], []
for pid, pdata in train_data.items():
    title = pdata.get("paper_title", "")
    for sk, sr in pdata.get("section_results", {}).items():
        fr = sr.get("final_report", {})
        content = fr.get("generated_content", "") or ""
        if len(content) < 10:
            iters = sr.get("iterations", [])
            if iters:
                best = max(iters, key=lambda x: x.get("evaluation", {}).get("overall_score", 0))
                content = best.get("generated_content", "") or ""
        if len(content) >= 10:
            texts.append(content[:1500])
            metas.append({"paper_id": pid, "section_key": sk, "research_topic": title,
                         "score": fr.get("source_eval_score", 5.0),
                         "reflection_summary": f"[{sk}] from {title}"})

print(f"  Encoding {len(texts)} vectors...")
all_embs = []
for i in range(0, len(texts), 6):
    batch = texts[i:i+6]
    resp = emb_client.embeddings.create(model=EMB_MODEL_NAME, input=batch)
    all_embs.extend([e.embedding for e in resp.data])

emb_matrix = np.array(all_embs).astype('float32')
faiss.normalize_L2(emb_matrix)
index = faiss.IndexFlatIP(EMB_DIM)
index.add(emb_matrix)
print(f"  [OK] FAISS ready | {index.ntotal} vectors x {EMB_DIM}d")

# Step 4: LLM client helper
def call_vllm(messages, temperature=0.7, max_tokens=2048):
    """Call local vLLM via requests (proven stable)"""
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = req_lib.post(VLLM_URL, json=payload, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy",
    }, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"vLLM HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()

# Test vLLM generation
print("\n[4] Testing vLLM generation...")
test_resp = call_vllm([{"role": "user", "content": "Say hello in 5 words."}], max_tokens=50)
print(f"  [OK] vLLM responds: '{test_resp[:80]}'")

# Step 5: Full inference on ONE topic
print("\n[5] Running FRAME Inference Pipeline...")
topic = "Deep Learning for CT-based Thymus Quantification using Automated Segmentation"

sections_to_generate = ["topic", "background", "related_work", "methodology", "result", "conclusion"]
results = {}

for sec_key in sections_to_generate:
    print(f"\n--- Generating [{sec_key}] ---")
    
    # RAG retrieval
    query_vec = np.array(emb_client.embeddings.create(model=EMB_MODEL_NAME, input=[topic]).data[0].embedding,
                          dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, min(5, index.ntotal))
    
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(metas):
            m = dict(metas[idx])
            m["similarity"] = float(score)
            candidates.append(m)
    
    print(f"  Retrieved {len(candidates)} candidates")
    
    # Simple context assembly (skip Filter/Integrator for first run - use top-3 directly)
    context_parts = []
    for c in candidates[:3]:
        ctx = c.get("reflection_summary", "")
        context_parts.append(ctx)
    
    reference_context = "\n\n".join(context_parts) if context_parts else ""
    
    # Generator prompt (simplified)
    system_msg = (
        "You are an expert academic writer specializing in medical research papers. "
        "Generate content that is rigorous, well-structured, and publication-quality. "
        "Respond with actual academic prose content only."
    )
    
    # Generator prompt
    ref_label = f"Reference materials from similar papers:\n{reference_context[:2000]}" if reference_context else ""
    user_prompt = f"""Write the [{sec_key}] section of an academic paper about:

Topic: {topic}

{ref_label}

Requirements:
- Write in formal academic English
- Be comprehensive but concise (~500-1000 words)
- Include relevant technical details
- This is for publication quality"""

    try:
        generated = call_vllm(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_prompt}],
            max_tokens=2048,
        )
        
        results[sec_key] = {
            "section_key": sec_key,
            "generated_content": generated,
            "retrieved_count": len(candidates),
            "method": "ours",
            "content_length": len(generated),
        }
        print(f"  [OK] {sec_key} ({len(generated)} chars)")
        
    except Exception as e:
        results[sec_key] = {"section_key": sec_key, "error": str(e)}
        print(f"  [FAIL] {sec_key}: {e}")
    
    time.sleep(0.5)

# Save result
out_path = os.path.join(PROJECT_ROOT, "results", "inference", "demo_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
final_result = {
    "research_topic": topic,
    "method": "ours",
    "sections": results,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

# Summary
print("\n" + "=" * 60)
print("  INFERENCE RESULT SUMMARY")
print("=" * 60)
ok_count = sum(1 for s in results.values() if "generated_content" in s and s["generated_content"])
total_chars = sum(s.get("content_length", 0) for s in results.values())
for sk, sr in results.items():
    content = sr.get("generated_content", "")
    err = sr.get("error", "")
    if content:
        print(f"\n  [OK] {sk}: {len(content)} chars")
        print(f"       {content[:150].replace(chr(10),' ')}...")
    elif err:
        print(f"\n  [FAIL] {sk}: {err[:100]}")

print(f"\n  Total: {ok_count}/6 sections OK | {total_chars} total chars")
print(f"  Saved: {out_path}")
print("=" * 60)
print("  Phase 4 Demo Complete!")
