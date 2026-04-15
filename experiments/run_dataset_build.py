#!/usr/bin/env python3
"""
Phase 2 入口: 数据集构建
运行 Extractor-Checker 迭代 + 三阶段过滤，生成结构化数据集

用法:
    python experiments/run_dataset_build.py
    # 或指定参数:
    python experiments/run_dataset_build.py --raw_dir data/raw/ --rounds 2 --target 100

环境变量:
    DASHSCOPE_API_KEY=your_key  (必需, 阿里云百炼)
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
import yaml


def load_raw_papers(raw_data_dir: str) -> list:
    """
    加载原始论文数据

    支持两种格式:
    1. data/raw/ 目录下的 JSON 文件（每行一个 {"id", "title", "text"}）
    2. data/raw/ 目录下直接放置的 .txt 文件（文件名=title, 内容=text）
    """
    papers = []
    raw_path = Path(raw_data_dir)

    if not raw_path.exists():
        logger.warning(f"原始数据目录不存在: {raw_data_dir}")
        return papers

    # 方式1: JSON 文件
    for json_file in raw_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                papers.extend(data)
            elif isinstance(data, dict):
                papers.append(data)
            logger.info(f"从 {json_file.name} 加载了 {len(data) if isinstance(data, list) else 1} 篇")
        except Exception as e:
            logger.error(f"读取 {json_file} 失败: {e}")

    # 方式2: txt/pdf 文件 (需要额外解析)
    if not papers:
        for txt_file in raw_path.glob("*.txt"):
            try:
                text = txt_file.read_text(encoding='utf-8')
                if len(text) > 100:
                    papers.append({
                        "id": txt_file.stem,
                        "title": txt_file.stem.replace('_', ' '),
                        "text": text,
                    })
            except Exception as e:
                logger.error(f"读取 {txt_file} 失败: {e}")

        # 尝试 PDF 解析 (可选)
        try:
            import pdfplumber
            for pdf_file in raw_path.glob("*.pdf"):
                try:
                    with pdfplumber.open(pdf_file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t:
                                text += t + "\n"
                    if len(text) > 200:
                        papers.append({
                            "id": pdf_file.stem,
                            "title": pdf_file.stem.replace('_', ' '),
                            "text": text,
                        })
                except Exception as e:
                    logger.warning(f"PDF 解析失败 {pdf_file}: {e}")
        except ImportError:
            pass

    # 去重
    seen_ids = set()
    unique_papers = []
    for p in papers:
        pid = p.get("id", "")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_papers.append(p)

    logger.info(f"\n总共加载 {len(unique_papers)} 篇原始论文")
    return unique_papers


def generate_demo_data(output_dir: str, n_samples: int = 20):
    """
    生成演示用的小规模测试数据（用于无真实数据时的快速验证）

    这不会产生真实的医学论文内容，仅用于验证流水线是否可运行。
    """
    from src.dataset.builder import stage_one_filter, DatasetBuilder

    demo_papers = []
    topics = [
        ("deep-learning-ct-imaging", "Deep Learning Approaches for CT-based Medical Image Analysis",
         'This study investigates the application of deep learning techniques to CT imaging analysis for clinical decision support. '
         'We collected 500 patient scans from three major hospitals including Massachusetts General Hospital, Johns Hopkins, and Mayo Clinic. '
         'Our proposed CNN architecture achieved 94.3% accuracy on lesion detection tasks, significantly outperforming traditional radiologist assessment (p < 0.001). '
         'The methodology employs a ResNet-50 backbone with attention mechanisms for spatial feature localization. '
         'Results demonstrate significant improvements in diagnostic speed and accuracy across all three testing sites. '
         'We further validate our approach on an independent cohort of 200 patients, achieving consistent performance with AUC of 0.96. '
         'The model shows particular strength in detecting early-stage pulmonary nodules that are often missed by human readers. '
         'Clinical integration tests indicate potential reduction in average diagnosis time from 48 hours to under 4 hours. '
         'Limitations include the need for multi-center prospective validation and handling of scanner-specific artifacts.'),

        ("nlp-clinical-notes", "NLP Methods for Clinical Note Information Extraction",
         'We present a transformer-based approach to extract structured information from unstructured clinical notes at scale. '
         'Using BERT fine-tuned on the MIMIC-III dataset containing over 2 million clinical notes, our model achieves F1=89.7% on entity recognition. '
         'The methodology incorporates domain-specific pre-training using biomedical literature and multi-task learning across entity types. '
         'Experimental results show consistent improvements across 5 hospital systems with varying documentation practices. '
         'Key entities extracted include medications, diagnoses, procedures, and temporal expressions with high precision. '
         'Error analysis reveals that abbreviations and non-standard terminology remain the primary failure modes. '
         'The system processes approximately 10,000 notes per hour on standard GPU infrastructure. '
         'Integration with existing EHR systems demonstrates feasibility for real-time clinical decision support applications.'),

        ("ai-drug-discovery", "Graph Neural Networks for Molecular Property Prediction",
         'This paper explores graph neural network architectures for accelerating drug discovery through molecular property prediction. '
         'We comprehensively evaluate multiple GNN variants on benchmark datasets including Tox21, MUV, HIV, and PCBA. '
         'The best performing model combining Graph Attention Networks with virtual node features achieves ROC-AUC of 0.847 on average. '
         'Key findings include the critical importance of edge feature engineering and transfer learning across molecular graphs of different sizes. '
         'Ablation studies demonstrate that message passing depth of 4-6 layers provides optimal trade-off between expressiveness and overfitting. '
         'We further propose a novel pre-training strategy using unlabeled molecular data that improves sample efficiency by 3x. '
         'Computational cost analysis shows each prediction requires only 5 milliseconds, enabling high-throughput screening applications. '
         'The approach is validated on retrospective analysis of 50 approved drugs, correctly predicting key properties in 82% of cases.'),

        ("federated-medical", "Federated Learning for Privacy-Preserving Medical AI",
         'A federated learning framework is developed for medical image classification while preserving patient privacy across institutions. '
         'Experiments with 10 simulated hospital networks demonstrate that FedProx consistently outperforms FedAvg by 4.2% in test accuracy. '
         'Privacy analysis confirms differential privacy guarantees under reasonable noise levels with epsilon budget below 1.0. '
         'The framework handles heterogeneous data distributions caused by varying patient demographics and equipment differences across sites. '
         'Communication efficiency is improved through gradient compression, reducing bandwidth requirements by 90% compared to baseline approaches. '
         'Convergence analysis proves theoretical guarantees under non-IID data assumptions common in healthcare settings. '
         'Real-world deployment considerations including client dropouts, asynchronous updates, and adversarial robustness are thoroughly evaluated. '
         'Results suggest federated approaches can enable collaborative AI development without centralizing sensitive medical data.'),

        ("radiology-report-gen", "Automated Radiology Report Generation from Medical Images",
         'We propose a novel vision-language model architecture for generating comprehensive radiology reports from chest X-ray images automatically. '
         'Our approach combines a DenseNet-121 visual encoder pretrained on ImageNet with a GPT-2 language decoder finetuned on radiology text corpora. '
         'Evaluated on the IU X-Ray dataset containing 7,470 radiology reports and associated images, CIDEr score reaches 1.42 and BLEU-4 achieves 0.38. '
         'Ablation studies confirm the critical value of anatomical attention modules that focus on clinically relevant regions. '
         'Comparison with board-certified radiologists shows the generated reports achieve comparable findings sections in 78% of test cases. '
         'The model successfully identifies critical findings including pneumothorax, consolidation, and cardiomegaly with sensitivity above 85%. '
         'Multi-task training with both report generation and disease classification auxiliary objectives provides mutual improvements. '
         'Clinical deployment would require integration with PACS systems and workflow optimization for radiologist review processes.'),
    ]

    import random
    random.seed(42)

    for i in range(n_samples):
        base = topics[i % len(topics)]
        demo_papers.append({
            "id": f"demo_{i+1:04d}",
            "title": f"[Demo] {base[1]} (sample {i+1})",
            "text": base[2] * (1 + i % 3),  # 变化长度
        })

    # 创建输出目录
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_dir / "demo_papers.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_papers, f, ensure_ascii=False, indent=2)

    logger.info(f"已生成 {len(demo_papers)} 条演示数据 -> {output_file}")
    return demo_papers


def main():
    parser = argparse.ArgumentParser(description="FRAME Phase 2: 数据集构建")
    parser.add_argument("--config", default="config/model_config.yaml", help="配置文件路径")
    parser.add_argument("--raw_dir", default="data/raw/", help="原始数据目录")
    parser.add_argument("--rounds", type=int, default=2, help="Extractor-Checker 迭代轮数 (默认2)")
    parser.add_argument("--threshold", type=float, default=6.0, help="质量门控阈值 (默认6.0)")
    parser.add_argument("--target", type=int, default=None, help="目标样本数 (默认使用全部)")
    parser.add_argument("--demo", action="store_true", help="生成演示数据并运行")
    parser.add_argument("--demo_n", type=int, default=20, help="演示数据数量")
    args = parser.parse_args()

    # 日志配置
    log_file = PROJECT_ROOT / "logs"
    log_file.mkdir(exist_ok=True)
    logger.add(str(log_file / "dataset_build.log"), rotation="50 MB", retention="7 days")
    logger.info("FRAME Phase 2: 数据集构建 开始")

    # 加载或生成数据
    if args.demo:
        logger.info(f"模式: 使用演示数据 ({args.demo_n} 样本)")
        papers = generate_demo_data(str(PROJECT_ROOT), n_samples=args.demo_n)
        args.raw_dir = str(PROJECT_ROOT / "data" / "raw")
    else:
        papers = load_raw_papers(args.raw_dir)

    if not papers:
        logger.error("没有找到任何论文数据! 请:")
        logger.error("   1. 将论文放入 data/raw/ 目录 (.json/.txt/.pdf 格式)")
        logger.error("   2. 或使用 --demo 参数生成演示数据快速测试")
        sys.exit(1)

    # 可选: 限制样本数
    if args.target and len(papers) > args.target:
        import random
        random.seed(42)
        papers = random.sample(papers, args.target)
        logger.info(f"随机采样至 {args.target} 篇")

    # 阶段 1: 基础过滤
    logger.info("\nStage 1: 基础质量筛选...")
    from src.dataset.builder import stage_one_filter, DatasetBuilder

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    ds_cfg = cfg.get('dataset', {})
    # Demo 模式自动降低长度阈值 (demo 文本约 800~1500 字符)
    min_len = int(ds_cfg.get('min_paper_length', 2000))
    if args.demo:
        min_len = min(min_len, 500)
        logger.info(f"Demo 模式: 降低最小长度阈值至 {min_len}")
    filtered_1 = stage_one_filter(
        papers,
        min_length=min_len,
    )

    if not filtered_1:
        logger.error("Stage 1 过滤后没有剩余样本!")
        sys.exit(1)

    # 运行 Extractor-Checker 迭代
    logger.info(f"\n开始 Extractor->Checker 迭代 (轮数={args.rounds})...")
    builder = DatasetBuilder(
        config_path=args.config,
        extraction_rounds=args.rounds,
        quality_threshold=args.threshold,
    )

    processed_dir = str(PROJECT_ROOT / ds_cfg.get("processed_dir", "data/processed/"))
    results = builder.run_batch(
        filtered_1,
        save_every=max(5, len(filtered_1) // 10),
        output_dir=processed_dir,
    )

    # 最终汇总
    logger.info("\n" + "=" * 60)
    logger.info("数据集构建完成!")
    logger.info(f"  输入:     {len(papers)} 篇")
    logger.info(f"  Stage 1:  {len(filtered_1)} 篇通过")
    logger.info(f"  成功提取: {sum(1 for r in results if r.extraction_success)} 篇")
    logger.info(f"  输出目录: {processed_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
