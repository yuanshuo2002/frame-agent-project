"""
Generator Agent - 生成论文章节内容
对应论文 Section 4.1: The Generator synthesizes manuscript sections
conditioned on the research topic acquisition
"""
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import LLMClient, create_client_from_role
from src.utils.prompts import get_generator_prompt, SECTION_CONFIG


class GeneratorAgent(BaseAgent):
    """
    Generator Agent - 根据研究主题和参考上下文生成论文章节

    在训练阶段: 基于原始论文内容 + 反馈历史生成改进版本
    在推理阶段: 基于研究主题 + 检索到的 Reflection Report 生成新章节
    """

    def __init__(self, client: Optional[LLMClient] = None):
        super().__init__("Generator", client or create_client_from_role("primary"))

    def run(
        self,
        section_key: str,
        research_topic: str,
        reference_context: str = "",
        **kwargs,
    ) -> str:
        """
        生成指定章节的内容

        Args:
            section_key: 章节标识 (topic/background/related_work/methodology/result/conclusion)
            research_topic: 研究主题/问题
            reference_context: 参考材料（训练时为原文，推理时为合并后的反馈报告）
            **kwargs: 额外参数（如 max_tokens 等）

        Returns:
            生成的章节文本内容
        """
        self.log_info(f"生成 [{section_key}] 章节...")

        prompt = get_generator_prompt(section_key, research_topic, reference_context)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic writer specializing in medical research papers. "
                    "Generate content that is rigorous, well-structured, and publication-quality. "
                    "Always respond with the actual content (not JSON), in natural academic prose."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            max_tokens = kwargs.get('max_tokens', None)
            response = self.client.chat(messages, temperature=0.7, max_tokens=max_tokens)
            self._record_call(success=True)
            self.log_info(f"✓ [{section_key}] 生成完成 ({len(response)} 字符)")
            return response.strip()

        except Exception as e:
            self.log_error(f"生成失败: {e}")
            self._record_call(success=False)
            raise


class TrainingGenerator(GeneratorAgent):
    """
    训练阶段的 Generator - 在 G→E→R 循环中使用
    输入包含前几轮的反馈信息用于迭代改进
    """

    def run_with_history(
        self,
        section_key: str,
        original_content: str,
        reflection_reports: list,
        iteration_round: int,
        research_topic: str = "",
        **kwargs,
    ) -> str:
        """
        结合历史反馈进行迭代生成

        Args:
            section_key: 章节标识
            original_content: 该章节的原始提取内容（作为基础）
            reflection_reports: 历史轮次的 Reflection Report 列表
            iteration_round: 当前迭代轮次
            research_topic: 研究主题
        """
        # 构建包含反馈历史的上下文
        history_context = ""
        if reflection_reports:
            history_context = "\n\n--- Previous Iteration Feedback ---\n"
            for i, report in enumerate(reflection_reports):
                history_context += f"\n[Iteration {i+1}]\n"
                if isinstance(report, dict):
                    history_context += f"Summary: {report.get('reflection_summary', 'N/A')}\n"
                    improvements = report.get('priority_improvements', [])
                    if improvements:
                        history_context += f"Priority Improvements:\n"
                        for imp in improvements:
                            history_context += f"  - {imp}\n"
                elif isinstance(report, str):
                    history_context += f"{report[:300]}\n"

            history_context += (
                "\nBased on ALL feedback above, generate an IMPROVED version that "
                f"addresses these issues. This is iteration round {iteration_round}."
            )

        # 合并参考上下文
        combined_reference = original_content
        if history_context:
            combined_reference = f"{original_content}\n\n{history_context}"

        return self.run(
            section_key=section_key,
            research_topic=research_topic or f"Medical research paper - {section_key}",
            reference_context=combined_reference,
            **kwargs,
        )
