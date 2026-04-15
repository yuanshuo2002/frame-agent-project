#!/usr/bin/env python3
"""
Phase 5 入口: 评估与对比实验
运行统计指标 + LLM-based 评估，生成完整对比报告

用法:
    # 评估已有的推理结果
    python experiments/run_evaluation.py --results results/inference/comparison_results.json

    # 完整流程 (训练→推理→评估)
    python experiments/run_evaluation.py --full_pipeline --demo
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="FRAME Phase 5: 评估 & 对比实验")
    parser.add_argument("--config", default="config/model_config.yaml")
    
    # 评估已有结果
    parser.add_argument("--results", type=str, default=None,
                        help="推理结果 JSON 路径")
    parser.add_argument("--test_data", type=str, default=None,
                        help="测试集数据路径")

    # 完整流水线
    parser.add_argument("--full_pipeline", action="store_true",
                        help="运行完整训练+推理+评估流程")
    parser.add_argument("--demo", action="store_true",
                        help="使用演示模式 (小规模快速验证)")

    # 参数
    parser.add_argument("--output_dir", type=str, default="results/evaluation/")
    parser.add_argument("--n_eval_samples", type=int, default=5,
                        help="评估最大样本数 (控制成本)")
    
    args = parser.parse_args()

    # 日志
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(str(log_dir / "evaluation.log"), rotation="50 MB", retention="7 days")

    logger.info("🚀 FRAME Phase 5: 评估阶段 开始")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.full_pipeline:
        # 完整流水线模式
        _run_full_pipeline(args, output_dir)
    elif args.results:
        # 仅评估已有结果
        _evaluate_existing_results(args, output_dir)
    else:
        # 默认: 尝试找已有结果来评估
        candidate_paths = [
            PROJECT_ROOT / "results" / "inference" / "comparison_results.json",
            PROJECT_ROOT / "results" / "inference" / "demo_results.json",
        ]
        for cp in candidate_paths:
            if cp.exists():
                args.results = str(cp)
                break
        
        if args.results:
            _evaluate_existing_results(args, output_dir)
        else:
            logger.info("未找到推理结果。请先运行:")
            logger.info("  python experiments/run_inference.py --demo")
            logger.info("或使用 --demo 运行完整演示")


def _run_full_pipeline(args, output_dir):
    """运行完整的 训练 → 推理 → 评估 流水线"""
    from src.inference.pipeline import FrameInferencePipeline, load_or_build_faiss_index

    logger.info("\n🔄 === 完整流水线模式 ===\n")

    # Step 1: 数据构建 (如果需要)
    train_path = PROJECT_ROOT / "data" / "processed" / "train.json"
    if not train_path.exists() or args.demo:
        if args.demo:
            logger.info("Step 1/4: 构建演示数据集...")
            import subprocess
            subprocess.run([
                sys.executable, str(PROJECT_ROOT / "experiments" / "run_dataset_build.py"),
                "--demo", "--demo_n", "10", "--rounds", "1"
            ], cwd=PROJECT_ROOT)

    # Step 2: 训练
    ckpt_path = PROJECT_ROOT / "checkpoints" / "training_results.json"
    if not ckpt_path.exists() or args.demo:
        if args.demo:
            logger.info("Step 2/4: 快速训练...")
            import subprocess
            subprocess.run([
                sys.executable, str(PROJECT_ROOT / "experiments" / "run_training.py"),
                "--rounds", "1", "--max_samples", "5"
            ], cwd=PROJECT_ROOT)

    # Step 3: 推理 + 对比
    logger.info("Step 3/4: 对比实验推理...")
    faiss_store = load_or_build_faiss_index(config_path=args.config)
    pipeline = FrameInferencePipeline(config_path=args.config, faiss_store=faiss_store)

    demo_topics = [
        "Deep Learning for CT-based Medical Image Analysis and Diagnosis",
        "NLP Methods for Clinical Note Information Extraction and Structuring",
        "Federated Learning Frameworks for Privacy-Preserving Medical AI Systems",
    ]

    comparison_results = pipeline.run_comparison_experiment(demo_topics[:args.n_eval_samples])

    inf_output = Path("results/inference")
    inf_output.mkdir(parents=True, exist_ok=True)
    comp_path = inf_output / "comparison_results.json"
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    # Step 4: 评估
    logger.info("Step 4/4: 评估对比结果...")
    test_path = PROJECT_ROOT / "data" / "processed" / "test.json"
    test_data = []
    if test_path.exists():
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

    benchmark_results = run_benchmark(
        comparison_results=comparison_results,
        test_data=test_data,
        config_path=args.config,
        n_samples=args.n_eval_samples,
        output_dir=output_dir,
    )

    report = format_and_save_report(benchmark_results, output_dir)
    print(report)


def _evaluate_existing_results(args, output_dir):
    """评估已有的推理结果"""
    results_path = args.results
    if not os.path.exists(results_path):
        logger.error(f"结果文件不存在: {results_path}")
        sys.exit(1)

    with open(results_path, 'r', encoding='utf-8') as f:
        comparison_results = json.load(f)

    # 加载测试数据
    test_data = []
    test_path = args.test_data or str(PROJECT_ROOT / "data" / "processed" / "test.json")
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        logger.info(f"加载测试集: {len(test_data)} 样本")

    logger.info(f"加载推理结果: {results_path}")
    
    benchmark_results = run_benchmark(
        comparison_results=comparison_results,
        test_data=test_data,
        config_path=args.config,
        n_samples=args.n_eval_samples,
        output_dir=output_dir,
    )

    report = format_and_save_report(benchmark_results, output_dir)
    print(report)


def run_benchmark(
    comparison_results: dict,
    test_data: list,
    config_path: str,
    n_samples: int,
    output_dir: Path,
) -> dict:
    """执行基准测试评估"""
    from src.evaluation.metrics import (
        run_benchmark_comparison,
        format_benchmark_report,
    )

    # 初始化 Embedding (用于统计指标)
    emb_model = None
    try:
        from src.utils.embedding import get_embedding_model
        emb_model = get_embedding_model(config_path)
    except Exception as e:
        logger.warning(f"Embedding 模型初始化失败，统计指标将跳过: {e}")

    # 限制评估样本数
    for method in list(comparison_results.keys()):
        papers = comparison_results.get(method, [])
        if len(papers) > n_samples:
            comparison_results[method] = papers[:n_samples]
            logger.info(f"{method}: 限制至 {n_samples} 个样本")

    results = run_benchmark_comparison(
        comparison_results=comparison_results,
        test_data=test_data,
        embedding_model=emb_model,
        config_path=config_path,
    )

    return results


def format_and_save_report(benchmark_results: dict, output_dir: Path) -> str:
    """格式化并保存报告"""
    from src.evaluation.metrics import format_benchmark_report
    
    report_text = format_benchmark_report(benchmark_results)

    # 保存文本报告
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"benchmark_report_{ts}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 保存原始 JSON 结果
    json_file = output_dir / "benchmark_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n📊 评估报告已保存:")
    logger.info(f"   文本版: {report_file}")
    logger.info(f"   数据版: {json_file}")

    return report_text


if __name__ == "__main__":
    main()
