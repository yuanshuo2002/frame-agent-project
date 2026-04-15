"""
Integrator - 合并多份 Reflection Report 为统一上下文
对应论文 4.2: we utilize an Integrator to consolidate and merge multiple pertinent reports.
This process is akin to using a larger batch size in Neural Network
rather than a batch size of one, facilitating the acquisition of more
balanced and comprehensive information.
"""
import os
import sys
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger
from src.utils.llm_client import LLMClient, create_client_from_role
from src.agents.base_agent import BaseAgent


INTEGRATOR_PROMPT_TEMPLATE = """Role: Knowledge Synthesis Specialist

Task: Merge and synthesize multiple reflection reports into a unified, coherent context that will guide the generation of a high-quality {section_name} section.

Target Research Topic: {research_topic}

{reports_block}

Instructions:
- Consolidate overlapping suggestions into unified guidance
- Remove contradictory or redundant information
- Prioritize the most impactful improvement insights (focus on patterns across reports)
- Produce a balanced, comprehensive synthesis

Output a merged context document in natural language ({target_length} words) that captures the collective wisdom from all reports. This will be used as reference material for generating the final paper section:
"""


class IntegratorAgent(BaseAgent):
    """
    Integrator Agent - 知识综合器

    将 Filter 筛选后的多份 Reflection Report 合并为一份统一的、
    无冗余的参考上下文，供 Generator 在推理阶段使用。

    核心思想：类比 NN 中增大 batch size，获得更全面的信息。
    """

    def __init__(self, client: Optional[LLMClient] = None):
        super().__init__("Integrator", client or create_client_from_role("primary"))
        self.max_reports = 5  # 一次最多合并的报告数量（防止上下文过长）

    def run(self, **kwargs) -> Any:
        """实现 BaseAgent 抽象接口，委托给 integrate"""
        return self.integrate(
            section_key=kwargs.get("section_key", ""),
            research_topic=kwargs.get("research_topic", ""),
            filtered_reports=kwargs.get("filtered_reports", []),
            target_length=kwargs.get("target_length", 600),
        )

    def integrate(
        self,
        section_key: str,
        research_topic: str,
        filtered_reports: List[Dict[str, Any]],
        target_length: int = 600,
    ) -> str:
        """
        合并多份报告为统一上下文

        Args:
            section_key: 目标章节标识
            research_topic: 研究主题
            filtered_reports: 经过 Filter 筛选后的报告列表
            target_length: 输出目标长度(词数)

        Returns:
            合并后的上下文文本
        """
        if not filtered_reports:
            self.log_info("没有可合并的报告")
            return ""

        # 截断到最大数量
        reports_to_merge = filtered_reports[:self.max_reports]
        n = len(reports_to_merge)

        self.log_info(f"合并 {n} 份报告 [{section_key}]...")

        # 构建报告块文本
        reports_block = self._format_reports(reports_to_merge)

        # 获取章节显示名
        from src.utils.prompts import SECTION_CONFIG
        sec_config = SECTION_CONFIG.get(section_key, SECTION_CONFIG["background"])
        sec_display_name = sec_config["display_name"]

        prompt = INTEGRATOR_PROMPT_TEMPLATE.format(
            section_name=sec_display_name,
            research_topic=research_topic,
            reports_block=reports_block,
            target_length=target_length,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at synthesizing multiple sources of feedback into "
                    "coherent guidance. Your output should be a well-organized reference document "
                    "that helps writers improve their academic work."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            merged_context = self.client.chat(
                messages,
                temperature=0.4,  # 低温度保证稳定性
                max_tokens=2048,
            )
            self._record_call(success=True)
            self.log_info(f"✓ 合并完成 | 输出={len(merged_context)} 字符 | 来源={n} 份报告")
            return merged_context.strip()

        except Exception as e:
            self.log_error(f"合并失败: {e}")
            self._record_call(success=False)
            # Fallback: 直接拼接摘要
            return self._fallback_concat(reports_to_merge)

    def _format_reports(self, reports: List[Dict[str, Any]]) -> str:
        """将多份报告格式化为结构化文本"""
        blocks = []
        for i, report in enumerate(reports):
            block = f"\n### Reflection Report {i+1}\n"

            section = report.get("section_key", "unknown")
            topic = report.get("research_topic", "")
            score = report.get("source_eval_score", "?")
            summary = report.get("reflection_summary", "")

            block += f"- **Section**: {section}\n"
            if topic:
                block += f"- **Original Topic**: {topic}\n"
            block += f"- **Source Eval Score**: {score}/5\n"
            if summary:
                block += f"\n**Summary**:\n{summary}\n"

            improvements = report.get("priority_improvements", [])
            if improvements:
                block += f"\n**Priority Improvements**:\n"
                for j, imp in enumerate(improvements, 1):
                    block += f"  {j}. {imp}\n"

            weaknesses = report.get("weaknesses", [])
            if weaknesses:
                block += f"\n**Key Weaknesses**:\n"
                for w in weaknesses[:3]:
                    if isinstance(w, dict):
                        issue = w.get("issue", "")
                        suggestion = w.get("suggestion", "")
                        block += f"  - **Issue**: {issue}\n"
                        block += f"    → **Suggestion**: {suggestion}\n"

            blocks.append(block)

        return "\n".join(blocks)

    @staticmethod
    def _fallback_concat(reports: List[Dict[str, Any]]) -> str:
        """降级方案：直接拼接各报告的 summary"""
        parts = []
        for i, r in enumerate(reports, 1):
            s = r.get("reflection_summary", r.get("raw_text", ""))[:300]
            parts.append(f"[Report {i}] {s}")
        return "\n\n".join(parts)
