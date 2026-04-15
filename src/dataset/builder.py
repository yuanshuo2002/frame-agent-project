"""
FRAME 数据集构建模块 - Phase 2
实现 Extractor-Checker 双 Agent 迭代精炼 + 三阶段过滤

对应论文 Section 3: Dataset Construction
"""
import os
import json
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger

# 导入工具模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.llm_client import LLMClient, create_client_from_role
from src.utils.prompts import (
    get_extractor_prompt, get_checker_prompt,
    SECTION_CONFIG,
)


@dataclass
class ExtractionResult:
    """单次提取结果"""
    section_key: str
    content: str           # 提取的内容
    score: float = 0.0     # Checker 评分 (0-10)
    reason: str = ""       # 评分理由
    iteration: int = 0     # 迭代轮次


@dataclass
class PaperExtraction:
    """单篇论文的完整提取结果"""
    paper_id: str
    paper_title: str
    full_text: str         # 原始全文
    sections: Dict[str, ExtractionResult] = field(default_factory=dict)
    extraction_success: bool = True
    error_message: str = ""


class ExtractorAgent:
    """
    Extractor Agent - 从论文中提取结构化章节内容
    对应论文 3.3 Data Extraction Agent
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or create_client_from_role("primary")
        self.section_keys = list(SECTION_CONFIG.keys())

    def extract(
        self,
        section_key: str,
        paragraph: str,
        previous_evaluations: str = "",
    ) -> Tuple[str, float, str]:
        """
        执行一次提取操作

        Returns:
            (extracted_content, score, reason) - 注意这里返回的是原始提取结果，
            实际评分由 Checker 完成，此处返回占位值
        """
        prompt = get_extractor_prompt(section_key, paragraph, previous_evaluations)

        messages = [
            {"role": "system", "content": "You are an expert academic paper analyst. Always respond in valid JSON format."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat_json(messages)
            # 提取对应 key 的内容
            content = response.get(section_key, "")
            if not content:
                # 尝试其他可能的 key
                for k, v in response.items():
                    if isinstance(v, str) and len(v) > 50:
                        content = v
                        break
            return content, 0.0, ""
        except Exception as e:
            logger.error(f"Extractor 失败 [{section_key}]: {e}")
            return "", 0.0, f"Extraction failed: {e}"


class CheckerAgent:
    """
    Checker Agent - 评估提取质量并给出分数和改进建议
    对应论文 3.3 Data Extraction Agent (Checker 部分)
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or create_client_from_role("evaluator")

    def check(
        self,
        section_key: str,
        original_paragraph: str,
        extracted_content: str,
    ) -> Tuple[float, str]:
        """
        评估提取内容的质量

        Returns:
            (score: float [0-10], reason: str)
        """
        if not extracted_content or len(extracted_content.strip()) < 20:
            return 1.0, "Extracted content is too short or empty."

        prompt = get_checker_prompt(section_key, original_paragraph, extracted_content)

        messages = [
            {"role": "system", "content": "You are a strict but fair academic quality evaluator. Always output valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat_json(messages)
            # DEBUG: 抓取 API 原始返回值，排查 score=0 问题
            logger.debug(f"  [{section_key}] Checker API 原始返回: {response}")
            score = float(response.get("score", 5.0))
            reason = response.get("reason", "No reason provided.")
            # 如果 score 异常低，打印警告便于排查
            if score < 1.0:
                logger.warning(f"  [{section_key}] Checker 给出极低分: score={score} | reason={reason[:200]} | raw_keys={list(response.keys())}")
            return max(0.0, min(10.0, score)), reason
        except Exception as e:
            logger.error(f"Checker 失败 [{section_key}]: {e}")
            return 3.0, f"Evaluation failed: {e}"


class DatasetBuilder:
    """
    数据集构建主控器 - 编排 Extractor → Checker 的 N 轮迭代 + 三阶段过滤

    对应论文 Section 3 全部内容
    """

    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        extraction_rounds: int = 2,
        quality_threshold: float = 6.0,
    ):
        self.config_path = config_path
        self.extraction_rounds = extraction_rounds  # N 轮迭代 (原文=3, 精简版=2)
        self.quality_threshold = quality_threshold   # 质量门控阈值

        self.extractor = ExtractorAgent()
        self.checker = CheckerAgent()

        # 长章节 vs 短章节策略 (论文 3.3)
        self.long_sections = ["related_work", "methodology", "result"]
        self.short_sections = ["topic", "background", "conclusion"]

        logger.info(f"DatasetBuilder 初始化完成 | 迭代轮数={extraction_rounds} | 阈值={quality_threshold}")

    def process_single_section(
        self,
        section_key: str,
        paragraph: str,
    ) -> ExtractionResult:
        """
        对单个章节执行 N 轮 Extractor → Checker 迭代

        Args:
            section_key: 章节标识符
            paragraph: 原始论文章节文本

        Returns:
            最优的 ExtractionResult
        """
        all_iterations: List[ExtractionResult] = []
        prev_eval_text = ""

        for round_idx in range(self.extraction_rounds):
            logger.info(f"  [{section_key}] 第 {round_idx+1}/{self.extraction_rounds} 轮...")

            # Step 1: Extractor 提取
            content, _, _ = self.extractor.extract(
                section_key, paragraph, prev_eval_text
            )

            if not content or len(content) < 10:
                logger.warning(f"  [{section_key}] 第 {round_idx+1} 轮提取为空，跳过")
                continue

            # Step 2: Checker 评估
            score, reason = self.checker.check(section_key, paragraph, content)

            result = ExtractionResult(
                section_key=section_key,
                content=content,
                score=score,
                reason=reason,
                iteration=round_idx + 1,
            )
            all_iterations.append(result)

            logger.info(f"  [{section_key}] Round {round_idx+1}: score={score:.2f}")

            # 构建历史记录供下一轮使用
            prev_eval_text = (
                f"Previous extraction (Round {round_idx+1}):\n"
                f"Content:\n{content[:500]}\n\n"
                f"Score: {score:.2f}\n"
                f"Reason: {reason}\n\n"
                f"Please improve based on this feedback."
            )

            # API 限流保护
            time.sleep(0.5)

        # 选择策略
        best = self._select_best_result(section_key, all_iterations)
        return best

    def _select_best_result(
        self,
        section_key: str,
        iterations: List[ExtractionResult],
    ) -> ExtractionResult:
        """
        根据章节类型选择最优迭代结果

        论文策略：
        - 长章节 (RelatedWork/Method/Result): 取最后一轮
        - 短章节 (Topic/Background/Conclusion): 质量门控选择
        """
        if not iterations:
            return ExtractionResult(
                section_key=section_key, content="", score=0.0,
                reason="No successful extractions", iteration=0
            )

        if section_key in self.long_sections:
            # 策略 1: 取最后一轮
            selected = iterations[-1]
            logger.debug(f"  [{section_key}] 长章节策略: 选择第 {selected.iteration} 轮 (最后)")
            return selected

        else:
            # 策略 2: 质量门控选择
            qualified = [
                it for it in iterations
                if it.score >= self.quality_threshold
            ]
            if qualified:
                selected = max(qualified, key=lambda x: x.score)
                logger.debug(f"  [{section_key}] 短章节策略(合格): 选择第 {selected.iteration} 轮, score={selected.score:.2f}")
                return selected
            else:
                # 没有合格的，选最高分的
                selected = max(iterations, key=lambda x: x.score)
                logger.debug(f"  [{section_key}] 短章节策略(全不合格): 最高分第 {selected.iteration} 轮, score={selected.score:.2f}")
                return selected

    def process_paper(self, paper_id: str, title: str, full_text: str) -> PaperExtraction:
        """
        处理一篇完整的论文，提取所有 6 个章节

        Args:
            paper_id: 论文唯一 ID
            title: 论文标题
            full_text: 论文全文文本

        Returns:
            PaperExtraction 包含所有章节的提取结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"处理论文: {title[:60]} (ID: {paper_id})")
        logger.info(f"{'='*60}")

        extraction = PaperExtraction(
            paper_id=paper_id,
            paper_title=title,
            full_text=full_text,
        )

        # 将全文作为各章节的输入（实际应用中可先做章节分割）
        # 这里简化处理：用全文作为每个章节的提取源
        for section_key in SECTION_CONFIG.keys():
            try:
                result = self.process_single_section(section_key, full_text)
                extraction.sections[section_key] = result
                logger.info(f"  ✓ [{section_key}] 完成 | score={result.score:.2f} | 内容长度={len(result.content)}")

                # 节流
                time.sleep(1)
            except Exception as e:
                logger.error(f"  ✗ [{section_key}] 异常: {e}")
                extraction.sections[section_key] = ExtractionResult(
                    section_key=section_key, content="", score=0.0,
                    reason=str(e), iteration=0
                )

        # 检查是否至少有部分章节成功提取
        success_count = sum(
            1 for s in extraction.sections.values()
            if s.content and len(s.content.strip()) >= 10  # 只要有内容就算通过 (demo 模式宽容)
        )
        if success_count < 2:  # 至少2章有内容即可
            extraction.extraction_success = False
            extraction.error_message = f"Only {success_count}/6 sections passed minimum threshold"

        return extraction

    def run_batch(
        self,
        papers: List[Dict[str, str]],
        save_every: int = 10,
        output_dir: str = "data/processed/",
    ) -> List[PaperExtraction]:
        """
        批量处理多篇论文

        Args:
            papers: 论文列表 [{"id": ..., "title": ..., "text": ...}, ...]
            save_every: 每隔多少篇保存一次中间结果
            output_dir: 输出目录

        Returns:
            所有论文的提取结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        all_results: List[PaperExtraction] = []

        total = len(papers)
        logger.info(f"\n开始批量处理 {total} 篇论文...")

        for idx, paper in enumerate(papers):
            result = self.process_paper(
                paper_id=paper["id"],
                title=paper["title"],
                full_text=paper["text"],
            )
            all_results.append(result)

            # 定期保存检查点
            if (idx + 1) % save_every == 0:
                self._save_checkpoint(all_results, output_dir, idx + 1)
                logger.progress = ((idx + 1) / total) * 100

        # 最终保存
        self._save_final(all_results, output_dir)

        # 统计
        success = sum(1 for r in all_results if r.extraction_success)
        logger.info(f"\n批量处理完成: {success}/{total} 成功 ({success/total*100:.1f}%)")

        return all_results

    def _save_checkpoint(self, results: List[PaperExtraction], output_dir: str, count: int):
        """保存中间检查点"""
        data = [self._extraction_to_dict(r) for r in results]
        path = os.path.join(output_dir, f"checkpoint_{count}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 检查点已保存: {path}")

    def _save_final(self, results: List[PaperExtraction], output_dir: str):
        """保存最终结果"""
        # 过滤掉失败的
        successful = [r for r in results if r.extraction_success]
        failed = [r for r in results if not r.extraction_success]

        # 保存成功的结果
        data = [self._extraction_to_dict(r) for r in successful]
        path = os.path.join(output_dir, "dataset_full.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 最终数据集已保存: {path} ({len(successful)} 篇)")

        if failed:
            fail_path = os.path.join(output_dir, "failed.json")
            fail_data = [{"id": r.paper_id, "title": r.paper_title, "error": r.error_message} for r in failed]
            with open(fail_path, 'w', encoding='utf-8') as f:
                json.dump(fail_data, f, ensure_ascii=False, indent=2)
            logger.info(f"⚠️ 失败记录已保存: {fail_path} ({len(failed)} 篇)")

    @staticmethod
    def _extraction_to_dict(extraction: PaperExtraction) -> Dict[str, Any]:
        """将 PaperExtraction 序列化为字典"""
        return {
            "paper_id": extraction.paper_id,
            "paper_title": extraction.paper_title,
            "sections": {
                key: {
                    "content": val.content,
                    "score": val.score,
                    "reason": val.reason,
                    "iteration": val.iteration,
                }
                for key, val in extraction.sections.items()
            },
            "extraction_success": extraction.extraction_success,
        }


# ============================================================
# 三阶段过滤器 (对应论文 3.2 第三段)
# ============================================================

def stage_one_filter(
    papers: List[Dict],
    min_length: int = 2000,
    llm_client: Optional[LLMClient] = None,
) -> List[Dict]:
    """
    第一阶段过滤: 基础质量筛选
    - 最小字数要求
    - 可扩展：期刊接受度、引用状态等
    """
    filtered = []
    for p in papers:
        text_len = len(p.get("text", ""))
        if text_len >= min_length:
            filtered.append(p)
        else:
            logger.debug(f"Stage1 过滤: '{p.get('title','?')[:40]}...' (长度={text_len})")
    logger.info(f"Stage 1 过滤: {len(papers)} → {len(filtered)} (最小长度={min_length})")
    return filtered


def stage_two_filter(
    extractions: List[Dict],
    min_sections_passed: int = 4,
    min_avg_score: float = 4.0,
) -> List[Dict]:
    """
    第二阶段过滤: 结构完整性检查
    - 至少有 N 个章节通过最低标准
    - 平均分不低于阈值
    """
    filtered = []
    for ext in extractions:
        sections = ext.get("sections", {})
        scores = [s.get("score", 0) for s in sections.values()]
        passed = sum(1 for s in scores if s >= 4.0)
        avg = sum(scores) / max(len(scores), 1)

        if passed >= min_sections_passed and avg >= min_avg_score:
            filtered.append(ext)
        else:
            logger.debug(f"Stage 2 过滤: '{ext.get('paper_title','?')[:40]}...' "
                        f"(通过={passed}/{min_sections_passed}, 均分={avg:.1f})")

    logger.info(f"Stage 2 过滤: {len(extractions)} → {len(filtered)} "
                f"(最少通过={min_sections_passed}, 最低均分={min_avg_score})")
    return filtered


def stage_three_filter(
    extractions: List[Dict],
    cutoff_date: str = "2024-09-01",
) -> Tuple[List[Dict], List[Dict]]:
    """
    第三阶段过滤: 时间划分 (训练集/测试集)
    使用截止日期避免数据泄露 (论文 5.1)

    Returns:
        (train_set, test_set)
    """
    from datetime import datetime

    # 如果没有日期字段，按比例随机分割
    has_date = any(e.get("publish_date") for e in extractions)

    if has_date:
        train = [e for e in extractions if e.get("publish_date", "") < cutoff_date]
        test = [e for e in extractions if e.get("publish_date", "") >= cutoff_date]
    else:
        # 无日期信息，按比例分割
        import random
        random.seed(42)
        shuffled = extractions.copy()
        random.shuffle(shuffled)
        split_point = int(len(shuffled) * 0.92)
        train = shuffled[:split_point]
        test = shuffled[split_point:]

    logger.info(f"Stage 3 划分: 训练集={len(train)}, 测试集={len(test)}")
    return train, test


def run_full_pipeline(papers: List[Dict], config_path: str = "config/model_config.yaml") -> str:
    """
    运行完整的 Stage 1→2→3 过滤流水线

    Returns:
        最终数据集文件路径
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get('dataset', {})

    # Stage 1
    filtered_1 = stage_one_filter(
        papers,
        min_length=int(ds_cfg.get('min_paper_length', 2000)),
    )

    # Stage 2 & 提取 (DatasetBuilder 内部包含提取逻辑)
    builder = DatasetBuilder(
        config_path=config_path,
        extraction_rounds=int(ds_cfg.get('extraction_rounds', 2)),
        quality_threshold=float(ds_cfg.get('quality_threshold', 6.0)),
    )
    extractions = builder.run_batch(
        filtered_1,
        save_every=10,
        output_dir=str(ds_cfg.get('processed_dir', 'data/processed/')),
    )

    # Stage 2: 后处理过滤
    raw_data_path = os.path.join(str(ds_cfg.get('processed_dir', 'data/processed/')), 'dataset_full.json')
    if os.path.exists(raw_data_path):
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            all_extractions = json.load(f)
        filtered_2 = stage_two_filter(all_extractions)

        # Stage 3: 划分训练/测试集
        train_set, test_set = stage_three_filter(filtered_2)

        # 分别保存
        processed_dir = str(ds_cfg.get('processed_dir', 'data/processed/'))
        for name, dataset in [("train", train_set), ("test", test_set)]:
            out_path = os.path.join(processed_dir, f"{name}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 {name} 集: {out_path} ({len(dataset)} 样本)")

        return raw_data_path

    return ""
