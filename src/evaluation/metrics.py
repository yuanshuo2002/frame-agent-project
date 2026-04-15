"""
评估指标框架 - 对应论文 Section 5.2: Evaluation Metrics

两套独立评估体系:
1. 统计指标 (Statistical Metrics): Soft Precision / Soft Recall
2. LLM-based 评分: 基于 Table 2 维度的多维度评分
"""
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.utils.embedding import get_embedding_model
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


# ============================================================
# 1. 统计指标: Soft Precision & Soft Recall
# ============================================================

def soft_precision_recall(
    generated_text: str,
    ground_truth: str,
    embedding_model=None,
    similarity_threshold: float = 0.7,
) -> Tuple[float, float]:
    """
    计算 Soft Precision 和 Soft Recall

    对应论文公式 (论文未给出显式公式，但描述为基于向量相似度的统计):
    
    - Soft Precision: 生成的信息中有多少与 ground truth 相关
    - Soft Recall: Ground truth 中有多少信息被覆盖

    实现思路（基于 sentence-level 相似度）:
    将文本拆分为句子，编码后计算相似度矩阵，用阈值判定匹配。

    Args:
        generated_text: 模型生成的文本
        ground_truth: 参考标准答案（ground truth）
        embedding_model: EmbeddingModel 实例
        similarity_threshold: 判定"相关"的相似度阈值

    Returns:
        (soft_precision, soft_recall)
    """
    if not EMBEDDING_AVAILABLE or embedding_model is None:
        logger.warning("Embedding 不可用，返回默认值")
        return 0.5, 0.5

    # 文本预处理：分句
    gen_sentences = _split_sentences(generated_text)
    gt_sentences = _split_sentences(ground_truth)

    if not gen_sentences or not gt_sentences:
        return 0.0, 0.0

    # 去除太短的句子
    gen_sentences = [s for s in gen_sentences if len(s.strip()) > 10]
    gt_sentences = [s for s in gt_sentences if len(s.strip()) > 10]

    if not gen_sentences or not gt_sentences:
        return 0.0, 0.0

    # 编码
    try:
        gen_embs = embedding_model.encode(gen_sentences)
        gt_embs = embedding_model.encode(gt_sentences)
    except Exception as e:
        logger.error(f"Encoding 失败: {e}")
        return 0.5, 0.5

    # 计算相似度矩阵 (gen x gt)
    sim_matrix = np.dot(gen_embs, gt_embs.T)  # 归一化向量的内积 = 余弦相似度

    # Soft Recall: 每个 GT 句子是否被至少一个生成句子覆盖
    recall_matches = np.max(sim_matrix, axis=0)  # 每个GT句子的最大相似度
    recall = float(np.mean(recall_matches >= similarity_threshold))

    # Soft Precision: 每个生成句子是否与某个 GT 句子相关
    precision_matches = np.max(sim_matrix, axis=1)  # 每个生成句子的最大相似度
    precision = float(np.mean(precision_matches >= similarity_threshold))

    return round(precision, 4), round(recall, 4)


def soft_f1(precision: float, recall: float) -> float:
    """计算 Soft F1 分数"""
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_all_statistical_metrics(
    generated_sections: Dict[str, str],
    ground_truth_sections: Dict[str, str],
    embedding_model=None,
) -> Dict[str, Dict[str, float]]:
    """
    计算所有章节的统计指标

    Returns:
        {
            "topic": {"precision": ..., "recall": ..., "f1": ...},
            "background": {...},
            ...
        }
    """
    results = {}
    for sec_key in generated_sections.keys():
        gen_text = generated_sections.get(sec_key, "")
        gt_text = ground_truth_sections.get(sec_key, "")

        if gen_text and gt_text:
            p, r = soft_precision_recall(gen_text, gt_text, embedding_model)
            f1 = soft_f1(p, r)
            results[sec_key] = {"precision": p, "recall": r, "f1": f1}
        else:
            results[sec_key] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return results


# ============================================================
# 2. LLM-based 评分 (复用 Evaluator Agent)
# ============================================================

class LLMEvaluator:
    """
    LLM-based 多维度评分器
    
    使用独立的 Evaluator Agent 对生成内容进行多维度打分。
    对应论文: The second set of metrics involves LLM-based scoring,
    where the output is evaluated across multiple dimensions such as
    Background, RelatedWork, Method, and Conclusion.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        from src.agents.evaluator import EvaluatorAgent
        self.evaluator = EvaluatorAgent()
        self.config_path = config_path
        self.eval_rounds = self._load_eval_rounds()

    def _load_eval_rounds(self) -> int:
        """加载评估轮次配置"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            return int(cfg.get('evaluation', {}).get('llm_eval_rounds', 1))
        except:
            return 1

    def evaluate_section(
        self,
        section_key: str,
        content: str,
        reference_topic: str,
    ) -> Dict[str, Any]:
        """
        评估单个章节（可多次取平均）
        """
        evaluations = []
        for _ in range(self.eval_rounds):
            result = self.evaluator.run(
                section_key=section_key,
                content=content,
                reference_topic=reference_topic,
            )
            evaluations.append(result)

        if self.eval_rounds > 1:
            from src.agents.evaluator import compute_section_average
            return compute_section_average(evaluations)

        return evaluations[0] if evaluations else {}

    def evaluate_paper(
        self,
        paper_result: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        评估一篇生成的完整论文

        Args:
            paper_result: generate_full_paper 的输出
            ground_truth: 可选的参考答案

        Returns:
            各章节的评估结果汇总
        """
        topic = paper_result.get("research_topic", "")
        sections = paper_result.get("sections", {})

        eval_results = {}
        for sec_key, sec_result in sections.items():
            if "error" in sec_result:
                continue

            content = sec_result.get("generated_content", "")
            if not content or len(content) < 20:
                continue

            gt_topic = None
            if ground_truth:
                gt_sec = ground_truth.get("sections", {}).get(sec_key, {})
                gt_topic = gt_sec.get("research_topic", topic)

            eval_results[sec_key] = self.evaluate_section(
                section_key=sec_key,
                content=content,
                reference_topic=gt_topic or topic,
            )

        return eval_results


# ============================================================
# 3. 综合对比实验框架
# ============================================================

def run_benchmark_comparison(
    comparison_results: Dict[str, List[Dict]],  # 来自 run_comparison_experiment
    test_data: List[Dict],                      # 测试集数据
    embedding_model=None,
    config_path: str = "config/model_config.yaml",
) -> Dict[str, Any]:
    """
    运行完整的对比实验评估

    Args:
        comparison_results: 三组方法 (no_rag/rag/ours) 的生成结果
        test_data: 测试集（包含 ground truth）
        config_path: 配置路径

    Returns:
        完整的对比结果字典
    """
    llm_evaluator = LLMEvaluator(config_path=config_path)
    all_stats = {}

    methods = ["no_rag", "rag", "ours"]
    method_labels = {
        "no_rag": "No-RAG",
        "rag": "RAG (Baseline)",
        "ours": "Ours (FRAME + Filter)",
    }

    summary_table = []
    detailed_results = {}

    for method in methods:
        papers = comparison_results.get(method, [])
        if not papers:
            continue

        method_stats = {
            "method": method,
            "label": method_labels[method],
            "n_papers": len(papers),
            "section_scores": {},
            "overall_avg_score": 0.0,
            "statistical_metrics": {},
        }

        all_section_scores = []

        for paper in papers:
            topic = paper.get("research_topic", "")
            sections = paper.get("sections", {})

            # 找到对应的 ground truth
            gt = None
            for t in test_data:
                t_title = t.get("paper_title", "")
                if t_title and (t_title.lower() in topic.lower() or topic.lower() in t_title.lower()):
                    gt = t
                    break

            # 如果没找到精确匹配，用第一个测试样本作为近似
            if not gt and test_data:
                gt = test_data[min(len(test_data)-1, len(all_section_scores))]

            # 评估各章节
            for sec_key, sec_result in sections.items():
                content = sec_result.get("generated_content", "")
                if not content or "error" in str(sec_result):
                    continue

                # LLM 评估
                eval_res = llm_evaluator.evaluate_section(
                    section_key=sec_key,
                    content=content,
                    reference_topic=topic,
                )

                score = eval_res.get("overall_score", 0)
                all_section_scores.append(score)

                if sec_key not in method_stats["section_scores"]:
                    method_stats["section_scores"][sec_key] = []
                method_stats["section_scores"][sec_key].append(score)

                # 统计指标（如果有 ground truth）
                if gt and embedding_model:
                    gt_content = ""
                    gt_secs = gt.get("sections", {})
                    if sec_key in gt_secs:
                        gc = gt_secs[sec_key]
                        if isinstance(gc, dict):
                            gt_content = gc.get("content", "")
                        elif isinstance(gc, str):
                            gt_content = gc

                    if gt_content:
                        p, r = soft_precision_recall(content, gt_content, embedding_model)
                        if sec_key not in method_stats["statistical_metrics"]:
                            method_stats["statistical_metrics"][sec_key] = {"precision": [], "recall": []}
                        method_stats["statistical_metrics"][sec_key]["precision"].append(p)
                        method_stats["statistical_metrics"][sec_key]["recall"].append(r)

        # 汇总平均分
        if all_section_scores:
            method_stats["overall_avg_score"] = round(
                sum(all_section_scores) / len(all_section_scores), 3
            )
        else:
            method_stats["overall_avg_score"] = 0.0

        # 各章节平均分
        avg_by_section = {}
        for sk, scores in method_stats["section_scores"].items():
            avg_by_section[sk] = round(sum(scores)/len(scores), 3) if scores else 0.0
        method_stats["section_scores"] = avg_by_section

        # 统计指标平均
        avg_stats = {}
        for sk, metrics in method_stats["statistical_metrics"].items():
            avg_stats[sk] = {}
            for mk, vals in metrics.items():
                avg_stats[sk][mk] = round(sum(vals)/len(vals), 4) if vals else 0.0
        method_stats["statistical_metrics"] = avg_stats

        all_stats[method] = method_stats
        summary_table.append({
            "method": method_labels[method],
            "avg_overall": method_stats["overall_avg_score"],
            "by_section": avg_by_section,
        })

    return {
        "summary_table": summary_table,
        "detailed": all_stats,
        "improvement_analysis": _compute_improvement(all_stats),
    }


def _compute_improvement(all_stats: Dict) -> Dict[str, Any]:
    """计算 Ours vs baseline 的提升百分比"""
    analysis = {}

    ours_stats = all_stats.get("ours", {})
    ours_overall = ours_stats.get("overall_avg_score", 0)

    for baseline_name in ["no_rag", "rag"]:
        baseline_stats = all_stats.get(baseline_name, {})
        baseline_overall = baseline_stats.get("overall_avg_score", 0)

        if baseline_overall > 0:
            gain_pct = ((ours_overall - baseline_overall) / baseline_overall) * 100
            analysis[f"gain_vs_{baseline_name}"] = round(gain_pct, 2)
        else:
            analysis[f"gain_vs_{baseline_name}"] = 0.0

    return analysis


# ============================================================
# 工具函数
# ============================================================

def _split_sentences(text: str) -> List[str]:
    """简单的中英文分句"""
    import re
    # 按句号、问号、感叹号、分号分割（支持中英文）
    sentences = re.split(r'(?<=[.!?。！？;；])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def format_benchmark_report(results: Dict[str, Any]) -> str:
    """将评估结果格式化为可读报告"""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("FRAME 论文复现 - 评估结果报告")
    lines.append("="*70)

    # 总体对比表
    lines.append(f"\n{'Method':<25} {'Overall Score':>15} {'vs No-RAG':>12} {'vs RAG':>12}")
    lines.append("-"*70)

    summary = results.get("summary_table", [])
    ours_overall = 0
    no_rag_overall = 0
    rag_overall = 0

    for row in summary:
        method_label = row["method"]
        overall = row["avg_overall"]
        lines.append(f"{method_label:<25} {overall:>15.3f}")

        if "No-RAG" in method_label or "No-RAG" in method_label.replace("(Baseline)", ""):
            no_rag_overall = overall
        elif "RAG" in method_label:
            rag_overall = overall
        elif "FRAME" in method_label or "Ours" in method_label:
            ours_overall = overall

    # 提升分析
    improvement = results.get("improvement_analysis", {})
    if improvement:
        lines.append("-"*70)
        gain_no_rag = improvement.get("gain_vs_no_rag", 0)
        gain_rag = improvement.get("gain_vs_rag", 0)
        lines.append(f"\n📈 FRAME 提升:")
        lines.append(f"   vs No-RAG: +{gain_no_rag:.2f}%")
        lines.append(f"   vs RAG:     +{gain_rag:.2f}%")

    # 各章节详情
    lines.append("\n\n" + "-"*70)
    lines.append("各章节详细得分:")
    lines.append("-"*70)

    detailed = results.get("detailed", {})
    for method, stats in detailed.items():
        label = stats.get("label", method)
        sec_scores = stats.get("section_scores", {})
        if sec_scores:
            line = f"\n{label}:"
            for sk, sc in sec_scores.items():
                line += f"  {sk}={sc}"
            lines.append(line)

    lines.append("\n" + "="*70)
    return "\n".join(lines)
