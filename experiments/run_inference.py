#!/usr/bin/env python3
"""
Phase 4 入口: 推理阶段 - 端到端论文生成

用法:
    # 基本用法: 生成单篇论文
    python experiments/run_inference.py --topic "Deep Learning for CT-based Thymus Quantification"

    # 对比实验 (No-RAG vs RAG vs Ours)
    python experiments/run_inference.py --mode comparison --topics_file data/test_topics.txt

    # 快速演示
    python experiments/run_inference.py --demo
"""
import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="FRAME Phase 4: 推理 - 论文生成")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--mode", choices=["single", "comparison", "demo"], default="demo",
                        help="运行模式: 单篇生成/对比实验/演示")
    
    # 单篇模式参数
    parser.add_argument("--topic", type=str, default=None, help="研究主题 (single 模式)")
    parser.add_argument("--method", choices=["no_rag", "rag", "ours"], default="ours",
                        help="推理方法 (默认=ours, 即完整FRAME流程)")

    # 对比实验模式参数
    parser.add_argument("--topics_file", type=str, default=None,
                        help="研究主题文件路径 (每行一个主题, comparison 模式)")
    parser.add_argument("--output_dir", type=str, default="results/inference/",
                        help="输出目录")

    args = parser.parse_args()

    # 日志
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(str(log_dir / "inference.log"), rotation="50 MB", retention="7 days")
    logger.add(sys.stderr, level="INFO")  # 确保控制台有输出

    # 加载配置 & 初始化流水线
    from src.inference.pipeline import FrameInferencePipeline, load_or_build_faiss_index
    from src.utils.embedding import get_embedding_model

    logger.info("🚀 FRAME Phase 4: 推理阶段 开始")

    # 初始化 FAISS (从训练结果构建)
    faiss_store = load_or_build_faiss_index(
        config_path=args.config,
    )

    pipeline = FrameInferencePipeline(
        config_path=args.config,
        faiss_store=faiss_store,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        _run_single(pipeline, args.topic, args.method, output_dir)

    elif args.mode == "comparison":
        _run_comparison(pipeline, args.topics_file, output_dir)

    else:
        _run_demo(pipeline, output_dir)


def _run_single(pipeline, topic: str, method: str, output_dir: Path):
    """单篇论文生成"""
    if not topic:
        logger.error("请通过 --topic 指定研究主题")
        sys.exit(1)

    logger.info(f"\n📝 生成论文 | 主题: {topic} | 方法: {method}")
    result = pipeline.generate_full_paper(research_topic=topic, method=method)

    # 保存
    safe_name = topic[:40].replace(" ", "_").replace("/", "_") + f"_{method}"
    out_path = output_dir / f"{safe_name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 论文已保存 → {out_path}")

    # 打印摘要
    print("\n" + "="*60)
    print(f"📄 生成的论文: {topic}")
    print("="*60)
    for sec_key, sec_result in result.get("sections", {}).items():
        content = sec_result.get("generated_content", "")
        if content and "error" not in sec_result:
            preview = content[:200].replace('\n', ' ')
            print(f"\n【{sec_key}】({len(content)} 字符):")
            print(f"  {preview}...")


def _run_comparison(pipeline, topics_file: str, output_dir: Path):
    """三组对比实验"""
    if not topics_file or not os.path.exists(topics_file):
        # 使用测试集的标题作为 topics
        test_path = PROJECT_ROOT / "data" / "processed" / "test.json"
        if test_path.exists():
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            topics = [t.get("paper_title", "") for t in test_data if t.get("paper_title")]
        else:
            # 使用内置 demo topics
            topics = [
                "Deep Learning Approaches for CT-based Medical Image Analysis",
                "NLP Methods for Clinical Note Information Extraction",
                "Federated Learning for Privacy-Preserving Medical AI",
                "Automated Radiology Report Generation from Medical Images",
                "Graph Neural Networks for Molecular Property Prediction",
            ]
    else:
        with open(topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]

    if not topics:
        logger.error("没有可用的研究主题!")
        sys.exit(1)

    logger.info(f"🔬 对比实验 | {len(topics)} 个主题 | 方法: No-RAG vs RAG vs Ours")

    results = pipeline.run_comparison_experiment(topics)

    # 保存完整结果
    out_path = output_dir / "comparison_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 对比实验结果已保存 → {out_path}")

    # 打印简要对比表
    _print_comparison_summary(results)


def _run_demo(pipeline, output_dir: Path):
    """快速演示模式"""
    demo_topics = [
        ("Deep Learning for CT-based Thymus Quantification using Automated Segmentation",
         "background"),
        ("Transformer-Based Models for Clinical Text Classification",
         "methodology"),
        ("Multi-modal Fusion for Medical Image Analysis",
         "conclusion"),
    ]

    logger.info("🎯 演示模式: 用 3 个示例主题测试各章节生成")

    all_results = {}
    for topic, focus_section in demo_topics:
        result = pipeline.generate_full_paper(topic, method="ours")
        safe_key = topic[:30].replace(" ", "_").lower()
        all_results[safe_key] = result

        print(f"\n{'─'*60}")
        print(f"主题: {topic}")
        print(f"{'─'*60}")
        for sk, sr in result.get("sections", {}).items():
            content = sr.get("generated_content", "")
            if content and "error" not in sr:
                print(f"\n【{sk}】({len(content)}字符):")
                print(content[:300] + "..." if len(content) > 300 else content)
        time_import = __import__('time')
        time_import.sleep(1)

    # 保存
    out_path = output_dir / "demo_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 演示结果已保存 → {out_path}")


def _print_comparison_summary(results: dict):
    """打印对比实验摘要表格"""
    methods = ["no_rag", "rag", "ours"]
    method_labels = {"no_rag": "No-RAG", "rag": "RAG", "ours": "Ours (FRAME)"}

    print("\n" + "="*70)
    print("📊 对比实验初步结果")
    print("="*70)
    print(f"{'方法':<20} {'成功数':>10} {'总生成':>10}")
    print("-"*70)

    for m in methods:
        papers = results.get(m, [])
        success = sum(1 for p in papers if "error" not in str(p))
        total_gen_sections = sum(
            len(p.get("sections", {})) for p in papers
        )
        print(f"{method_labels[m]:<20} {success:>10} {total_gen_sections:>10}")

    print("="*70)
    print("\n💡 完整评估请运行: python experiments/run_evaluation.py")
