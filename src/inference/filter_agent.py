"""
Filter Agent - RAG 检索后的二次相关性筛选
对应论文 4.2: we introduce a model known as the Filter, which acts as a gatekeeper
by eliminating reports that appear proximal in the vector space but are not truly relevant
"""
import os
import sys
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger
from src.utils.llm_client import LLMClient, create_client_from_role
from src.agents.base_agent import BaseAgent


FILTER_PROMPT_TEMPLATE = """Role: Relevance Gatekeeper

Task: Determine whether the following Reflection Report is truly relevant and useful for generating a paper on the given research topic.

Research Topic: {research_topic}

Reflection Report (candidate):
```
{report_content}
```

Evaluation Criteria:
1. **Topic Alignment**: Does the report address challenges/methods/findings directly related to the topic?
2. **Actionability**: Can the suggestions in this report actually help improve paper generation?
3. **Specificity**: Are the suggestions concrete enough to be useful?

Output JSON (ONLY valid JSON, no other text):
{{
    "is_relevant": true/false,
    "relevance_score": <0.0-1.0>,
    "reason": "<brief explanation of decision>"
}}

Only reports with relevance_score >= 0.6 should be considered relevant.
"""


class FilterAgent(BaseAgent):
    """
    Filter Agent - 门卫/守门人角色

    在 FAISS 检索后，对候选报告进行 LLM 级别的相关性判断，
    过滤掉"向量空间接近但实际不相关"的噪声报告。
    """

    def __init__(self, client: Optional[LLMClient] = None):
        super().__init__("Filter", client or create_client_from_role("evaluator"))
        self.relevance_threshold = 0.6  # 论文隐含阈值

    def run(self, **kwargs) -> Any:
        """实现 BaseAgent 抽象接口，委托给 filter_single"""
        return self.filter_single(
            research_topic=kwargs.get("research_topic", ""),
            candidate_report=kwargs.get("candidate_report", {}),
        )

    def filter_single(
        self,
        research_topic: str,
        candidate_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        对单个候选报告进行过滤判断

        Returns:
            {
                "is_relevant": bool,
                "relevance_score": float,
                "reason": str,
                "original_report": Dict,   # 原始报告（透传）
            }
        """
        # 构建报告文本摘要
        report_text = self._summarize_report(candidate_report)

        prompt = FILTER_PROMPT_TEMPLATE.format(
            research_topic=research_topic,
            report_content=report_text[:2000],  # 截断防止过长
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict but fair relevance evaluator. Your job is to determine "
                    "whether a reflection report is genuinely helpful for writing about a specific "
                    "research topic. Be conservative - only pass truly relevant reports."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            result = self.client.chat_json(messages, temperature=0.2)
            is_relevant = result.get("is_relevant", False)
            score = float(result.get("relevance_score", 0.0))
            reason = result.get("reason", "")

            # 如果 score >= threshold 但 is_relevant 为 False，以 score 为准
            if not is_relevant and score >= self.relevance_threshold:
                is_relevant = True

            output = {
                "is_relevant": is_relevant,
                "relevance_score": score,
                "reason": reason,
                "original_report": candidate_report,
            }

            status = "✓ 通过" if is_relevant else "✗ 过滤"
            self.log_info(f"{status} | score={score:.2f} | {reason[:80]}")
            self._record_call(success=True)

        except Exception as e:
            self.log_error(f"Filter 判断失败: {e}")
            self._record_call(success=False)
            # 出错时保守处理：保留报告但标记低分
            output = {
                "is_relevant": False,
                "relevance_score": 0.0,
                "reason": f"Filter error: {e}",
                "original_report": candidate_report,
            }

        return output

    def batch_filter(
        self,
        research_topic: str,
        candidates: List[Dict[str, Any]],
        min_keep: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        批量过滤，保证至少保留 min_keep 个结果

        Args:
            research_topic: 研究主题
            candidates: 候选报告列表 (来自 FAISS 检索结果)
            min_keep: 最少保留数量

        Returns:
            过滤后的报告列表
        """
        if not candidates:
            return []

        filtered_results = []
        for cand in candidates:
            result = self.filter_single(research_topic, cand)
            if result["is_relevant"]:
                filtered_results.append(result["original_report"])

        # 保证至少有 min_keep 个结果（按相似度排序补齐）
        if len(filtered_results) < min_keep:
            # 从被过滤的结果中按分数最高的补回
            remaining = [c for c in candidates if c not in filtered_results]
            # 假设候选列表已按相似度降序排列
            for r in remaining[:min_keep - len(filtered_results)]:
                if r not in filtered_results:
                    filtered_results.append(r)
                    self.log_info(f"⚠ 补充保留 (数量不足)")

        self.log_info(f"批量过滤完成: {len(candidates)} → {len(filtered_results)}")
        return filtered_results

    def _summarize_report(self, report: Dict[str, Any]) -> str:
        """将报告字典压缩为可读文本摘要"""
        parts = []
        section = report.get("section_key", "unknown")
        topic = report.get("research_topic", "")
        summary = report.get("reflection_summary", "")

        parts.append(f"[Section: {section}]")
        if topic:
            parts.append(f"Original Topic: {topic}")
        if summary:
            parts.append(f"Summary: {summary}")

        improvements = report.get("priority_improvements", [])
        if improvements:
            imp_str = "; ".join(improvements[:5])
            parts.append(f"Priority Improvements: {imp_str}")

        weaknesses = report.get("weaknesses", [])
        if weaknesses:
            w_texts = []
            for w in weaknesses[:3]:
                if isinstance(w, dict):
                    w_texts.append(f"- {w.get('issue', '')}: {w.get('suggestion', '')}")
                else:
                    w_texts.append(str(w))
            if w_texts:
                parts.append(f"Weaknesses:\n" + "\n".join(w_texts))

        return "\n".join(parts) or json.dumps(report, ensure_ascii=False)[:1000]


import json
