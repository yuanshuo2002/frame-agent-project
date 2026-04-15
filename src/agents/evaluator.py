"""
Evaluator Agent - 多维度质量评估
对应论文 Section 4.1: The Evaluator conducts multi-dimensional quality assessments
using the evaluation metrics defined in Table 2, producing quantitative scores (1-5 scale)
"""
import json
import os
import sys
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import LLMClient, create_client_from_role
from src.utils.prompts import get_evaluator_prompt


class EvaluatorAgent(BaseAgent):
    """
    Evaluator Agent - 对生成内容进行多维度质量评分

    评估维度来自论文 Table 2，每个章节有 6 个维度，每维度 1-5 分（0.1 步进）
    """

    def __init__(self, client: Optional[LLMClient] = None):
        super().__init__("Evaluator", client or create_client_from_role("evaluator"))

        # 加载评估维度定义
        self._dimensions_cache: Dict[str, str] = {}

    def _get_dimensions_text(self, section_key: str) -> str:
        """获取指定章节的评估维度文本描述"""
        if section_key in self._dimensions_cache:
            return self._dimensions_cache[section_key]

        # 动态从 YAML 加载维度信息
        import yaml

        config_path = "config/eval_dimensions.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            sections_cfg = cfg.get('sections', {})
            sec_cfg = sections_cfg.get(section_key, {})
            dims = sec_cfg.get('evaluation_dimensions', [])

            dim_lines = []
            for d in dims:
                name = d.get('name', '')
                desc = d.get('description', '')
                r = d.get('score_range', [1, 5])
                dim_lines.append(f"- **{name}** ({r[0]}-{r[1]}): {desc}")

            result = "\n".join(dim_lines)
            self._dimensions_cache[section_key] = result
            return result
        except Exception:
            return "Evaluate on completeness, relevance, organization, style, and depth."

    def _build_dimension_keys(self, section_key: str) -> List[str]:
        """获取该章节的维度 key 列表（用于 JSON 解析）"""
        import yaml
        config_path = "config/eval_dimensions.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        sec_cfg = cfg.get('sections', {}).get(section_key, {})
        return [d.get('name', f'dim_{i}') for i, d in enumerate(sec_cfg.get('evaluation_dimensions', []))]

    def run(
        self,
        section_key: str,
        content: str,
        reference_topic: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        对内容执行多维度评估

        Args:
            section_key: 章节标识
            content: 待评估的内容文本
            reference_topic: 参考研究主题（用于相关性检查）

        Returns:
            评估结果字典:
            {
                "dimension_scores": {"dim_name": {"score": float, "reason": str}, ...},
                "overall_score": float,
                "summary": str,
                "raw_response": str  # 原始响应
            }
        """
        self.log_info(f"评估 [{section_key}] 章节 ({len(content)} 字符)...")

        dimensions_info = self._get_dimensions_text(section_key)
        dim_keys = self._build_dimension_keys(section_key)

        prompt = get_evaluator_prompt(section_key, content, reference_topic, dimensions_info)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic reviewer with decades of experience evaluating "
                    "medical research papers. Be objective, thorough, and fair. "
                    "ALWAYS output valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw_response = self.client.chat_json(
                messages,
                temperature=0.3,  # 低温度保证稳定性
                max_tokens=2048,
            )

            # 解析并规范化结果
            dimension_scores = raw_response.get("dimension_scores", {})
            overall_score = float(raw_response.get("overall_score", 3.0))
            summary = raw_response.get("summary", "")

            # 确保所有维度都有分数
            for dk in dim_keys:
                if dk not in dimension_scores:
                    dimension_scores[dk] = {"score": 3.0, "reason": "Default score - not evaluated"}
                elif isinstance(dimension_scores[dk], (int, float)):
                    dimension_scores[dk] = {
                        "score": float(dimension_scores[dk]),
                        "reason": "Score provided without detailed reason"
                    }

            result = {
                "section_key": section_key,
                "dimension_scores": dimension_scores,
                "overall_score": overall_score,
                "summary": summary,
                "dim_keys": dim_keys,
            }

            self._record_call(success=True)
            self.log_info(f"✓ [{section_key}] 评估完成 | 总分={overall_score:.2f}")
            return result

        except Exception as e:
            self.log_error(f"评估失败: {type(e).__name__}: {e}")
            self._record_call(success=False)
            # 返回默认低分结果
            return {
                "section_key": section_key,
                "dimension_scores": {dk: {"score": 2.0, "reason": f"Evaluation failed: {e}"} for dk in (dim_keys or ["default"])},
                "overall_score": 2.0,
                "summary": f"Evaluation failed: {e}",
                "dim_keys": dim_keys or ["default"],
            }


def compute_section_average(evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算多次评估的平均值 (对应论文: three independent assessments)

    Args:
        evaluations: 同一内容的多次独立评估结果列表

    Returns:
        平均后的评估结果
    """
    if not evaluations:
        return {}

    n = len(evaluations)
    avg_result = {"overall_score": 0.0, "dimension_scores": {}}

    total_overall = sum(ev.get("overall_score", 0) for ev in evaluations)
    avg_result["overall_score"] = round(total_overall / n, 2)

    # 获取所有维度名
    all_dim_keys = set()
    for ev in evaluations:
        ds = ev.get("dimension_scores", {})
        all_dim_keys.update(ds.keys())

    for dk in all_dim_keys:
        scores = []
        for ev in evaluations:
            ds = ev.get("dimension_scores", {}).get(dk, {})
            if isinstance(ds, dict):
                scores.append(ds.get("score", 0))
            elif isinstance(ds, (int, float)):
                scores.append(float(ds))

        if scores:
            avg_result["dimension_scores"][dk] = {
                "score": round(sum(scores) / len(scores), 2),
                "reason": f"Averaged over {n} evaluations",
            }

    return avg_result
