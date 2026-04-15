#!/usr/bin/env python3
"""
Phase 3 入口: FRAME Agent 训练阶段
运行 Generator → Evaluator → Reflector 的 N 轮闭环迭代，生成 Reflection Reports

用法:
    python experiments/run_training.py
    python experiments/run_training.py --dataset data/processed/train.json --rounds 2

输出:
    checkpoints/training_results.json   (所有训练结果)
    data/faiss_index/frame_index.*       (FAISS 向量库)
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
    parser = argparse.ArgumentParser(description="FRAME Phase 3: Agent 训练 (G→E→R 循环)")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--dataset", default=None, help="训练集 JSON 路径")
    parser.add_argument("--rounds", type=int, default=2, help="G-E-R 迭代轮数")
    parser.add_argument("--max_samples", type=int, default=None, help="最大处理样本数 (用于快速测试)")
    parser.add_argument("--sections", nargs="+", default=None,
                        help="只处理指定章节 (如: background methodology conclusion)")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已有结果的样本")
    args = parser.parse_args()

    # 日志
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(str(log_dir / "training.log"), rotation="50 MB", retention="7 days")

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('training', {})

    # 自动查找可用的训练数据集 (优先 train.json, 其次 dataset_full.json)
    processed_dir = PROJECT_ROOT / ds_cfg.get("processed_dir", "data/processed/")
    
    if args.dataset:
        dataset_path = Path(args.dataset)
    elif (processed_dir / "train.json").exists():
        dataset_path = processed_dir / "train.json"
    elif (processed_dir / "dataset_full.json").exists():
        dataset_path = processed_dir / "dataset_full.json"
        logger.info(f"未找到 train.json, 回退使用 dataset_full.json ({len(json.load(open(dataset_path, 'r', encoding='utf-8')))} 样本)")
    else:
        logger.error(f"训练数据集不存在: {processed_dir}")
        logger.error("请先运行: python experiments/run_dataset_build.py --demo")
        sys.exit(1)

    # 加载训练数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    logger.info(f"加载训练数据: {len(train_data)} 样本 -> {dataset_path}")

    # 限制样本数
    if args.max_samples and len(train_data) > args.max_samples:
        import random; random.seed(42)
        train_data = random.sample(train_data, args.max_samples)
        logger.info(f"限制至 {args.max_samples} 个样本 (快速测试模式)")

    # 目标章节
    all_sections = ["topic", "background", "related_work", "methodology", "result", "conclusion"]
    target_sections = args.sections or all_sections

    # 初始化训练循环
    from src.agents.reflector import FrameTrainingLoop

    loop = FrameTrainingLoop(
        config_path=args.config,
        iteration_rounds=args.rounds,
    )

    # 输出目录
    ckpt_dir = PROJECT_ROOT / train_cfg.get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    output_file = ckpt_dir / "training_results.json"

    # 加载已有进度
    existing_results = {}
    if args.skip_existing and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        logger.info(f"已加载 {len(existing_results)} 条已有结果")

    # 主循环
    total_tasks = len(train_data) * len(target_sections)
    completed = 0

    for sample_idx, sample in enumerate(train_data):
        paper_id = sample.get("paper_id", f"sample_{sample_idx}")
        paper_title = sample.get("paper_title", paper_id)
        sections_data = sample.get("sections", {})

        logger.info(f"\n{'#'*60}")
        logger.info(f"[{sample_idx+1}/{len(train_data)}] 处理论文: {paper_title[:50]}...")
        logger.info(f"{'#'*60}")

        if paper_id not in existing_results:
            existing_results[paper_id] = {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_results": {},
            }

        for section_key in target_sections:
            task_key = f"{paper_id}_{section_key}"

            # 检查是否已完成
            if args.skip_existing and task_key in [k for k in existing_results[paper_id].get("section_results", {}).keys()]:
                logger.info(f"  ⏭️ [{section_key}] 已完成，跳过")
                continue

            # 获取该章节的原始内容
            sec_info = sections_data.get(section_key, {})
            original_content = ""
            if isinstance(sec_info, dict):
                original_content = sec_info.get("content", "")
            elif isinstance(sec_info, str):
                original_content = sec_info

            if not original_content or len(original_content) < 20:
                logger.warning(f"  ⚠️ [{section_key}] 原始内容为空或过短，跳过")
                continue

            try:
                # 执行 G→E→R 训练循环
                result = loop.process_sample(
                    section_key=section_key,
                    original_content=original_content,
                    research_topic=paper_title,
                )

                existing_results[paper_id]["section_results"][section_key] = result
                completed += 1

                # 定期保存
                save_interval = int(train_cfg.get('save_every_n_samples', 10))
                if completed % save_interval == 0:
                    _save_checkpoint(existing_results, output_file)
                    pct = completed / total_tasks * 100
                    logger.info(f"💾 进度检查点 | {completed}/{total_tasks} ({pct:.1f}%)")

            except Exception as e:
                logger.error(f"  ❌ [{section_key}] 异常: {e}")
                existing_results[paper_id]["section_results"][section_key] = {
                    "error": str(e),
                    "section_key": section_key,
                }

            # API 节流
            time_import = __import__('time')
            time_import.sleep(1)

    # 最终保存
    _save_checkpoint(existing_results, output_file)

    # 统计
    success_count = sum(
        1 for pid, pdata in existing_results.items()
        for sk, sr in pdata.get("section_results", {}).items()
        if "error" not in sr
    )
    total_count = sum(
        1 for pid, pdata in existing_results.items()
        for _ in pdata.get("section_results", {}).items()
    )

    logger.info("\n" + "="*60)
    logger.info("🎉 FRAME 训练完成!")
    logger.info(f"  总任务: {total_count}")
    logger.info(f"  成功:   {success_count} ({success_count/max(total_count,1)*100:.1f}%)")
    logger.info(f"  结果文件: {output_file}")
    logger.info("="*60)


def _save_checkpoint(data, path: Path):
    """原子性保存检查点"""
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Windows 下 rename 不能覆盖已存在的文件，需要先删除目标
    if path.exists():
        path.unlink()
    tmp_path.rename(path)


if __name__ == "__main__":
    main()
