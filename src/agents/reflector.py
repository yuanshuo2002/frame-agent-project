"""
Reflector Agent - 将评估结果转化为结构化改进建议
对应论文 Section 4.1: The Reflector translates Evaluator feedback into
structured reflection reports by mapping criticism to specific suggestion dimensions
"""
import json
import os
import sys
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.base_agent import BaseAgent
from src.agents.generator import TrainingGenerator
from src.agents.evaluator import EvaluatorAgent
from src.utils.llm_client import LLMClient, create_client_from_role
from src.utils.prompts import get_reflector_prompt


class ReflectorAgent(BaseAgent):
    """
    Reflector Agent - 将 Evaluator 的评分结果转化为具体的、可操作的改进建议

    这是 FRAME 方法的关键组件之一，它将抽象的数字分数映射为具体写作指导。
    输出的 Reflection Report 将被存储到 FAISS 中供推理阶段检索使用。
    """

    def __init__(self, client: Optional[LLMClient] = None):
        super().__init__("Reflector", client or create_client_from_role("evaluator"))

        # 加载改进建议映射表
        self._rec_mapping_cache: Dict[str, str] = {}

    def _get_recommendation_mapping(self, section_key: str) -> str:
        """获取章节对应的建议维度映射文本"""
        if section_key in self._rec_mapping_cache:
            return self._rec_mapping_cache[section_key]

        import yaml
        config_path = "config/eval_dimensions.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            mapping = cfg.get('recommendation_mapping', {}).get(section_key, {})
            lines = [f"- **{k}**: {v}" for k, v in mapping.items()]
            result = "\n".join(lines) if lines else "General improvement suggestions."
            self._rec_mapping_cache[section_key] = result
            return result
        except Exception:
            return "Provide specific, actionable suggestions for improvement."

    def run(
        self,
        section_key: str,
        evaluation_result: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        将评估结果转化为 Reflection Report

        Args:
            section_key: 章节标识
            evaluation_result: Evaluator Agent 的输出

        Returns:
            Reflection Report (字典格式):
            {
                "section_key": str,
                "strengths": [...],
                "weaknesses": [{"dimension", "score", "issue", "suggestion"}, ...],
                "priority_improvements": [...],
                "reflection_summary": str,
                "raw_text": str  # 完整报告文本（用于 Embedding）
            }
        """
        self.log_info(f"反思 [{section_key}] 章节...")

        # 构建评估结果的文本描述
        eval_text = self._format_evaluation(evaluation_result)

        rec_mapping = self._get_recommendation_mapping(section_key)

        prompt = get_reflector_prompt(section_key, eval_text, rec_mapping)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert writing coach specializing in academic medical research. "
                    "Your job is to translate numerical evaluation scores into concrete, "
                    "actionable improvement guidance. Be specific and practical."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw_response = self.client.chat_json(
                messages,
                temperature=0.5,
                max_tokens=2048,
            )

            report = {
                "section_key": section_key,
                "source_eval_score": evaluation_result.get("overall_score", 0),
                "strengths": raw_response.get("strengths", []),
                "weaknesses": raw_response.get("weaknesses", []),
                "priority_improvements": raw_response.get("priority_improvements", []),
                "reflection_summary": raw_response.get("reflection_summary", ""),
                "raw_text": json.dumps(raw_response, ensure_ascii=False),
            }

            self._record_call(success=True)
            self.log_info(f"✓ [{section_key}] 反思完成 | 弱点={len(report['weaknesses'])} 个")
            return report

        except Exception as e:
            self.log_error(f"反思失败: {e}")
            self._record_call(success=False)
            # 返回基础报告
            return {
                "section_key": section_key,
                "source_eval_score": evaluation_result.get("overall_score", 0),
                "strengths": ["Content was generated"],
                "weaknesses": [{
                    "dimension": "general",
                    "score": evaluation_result.get("overall_score", 3.0),
                    "issue": f"Reflection failed: {e}",
                    "suggestion": "Review content for completeness and clarity",
                }],
                "priority_improvements": ["Review overall quality"],
                "reflection_summary": f"Reflection generation failed: {e}",
                "raw_text": "",
            }

    def _format_evaluation(self, eval_result: Dict) -> str:
        """将 Evaluator 结果格式化为可读文本"""
        lines = []
        lines.append(f"**Overall Score**: {eval_result.get('overall_score', 'N/A')}/5.0")
        lines.append("")
        lines.append("**Dimension Breakdown:**")
        dim_scores = eval_result.get("dimension_scores", {})
        for dim_name, dim_data in dim_scores.items():
            if isinstance(dim_data, dict):
                score = dim_data.get("score", "N/A")
                reason = dim_data.get("reason", "")
                lines.append(f"- {dim_name}: {score}/5 | {reason}")
            elif isinstance(dim_data, (int, float)):
                lines.append(f"- {dim_name}: {dim_data}/5")

        summary = eval_result.get("summary", "")
        if summary:
            lines.append("")
            lines.append(f"**Evaluator Summary**: {summary}")

        return "\n".join(lines)


class FrameTrainingLoop:
    """
    FRAME 训练循环主控器
    编排 Generator → Evaluator → Reflector 的 N 轮闭环迭代

    对应论文 Section 4.1 (Agent Training Stage) + Figure 1 上半部分
    """

    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        iteration_rounds: int = 2,
    ):
        self.iteration_rounds = iteration_rounds  # N 轮训练循环 (原文=3, 精简版=2)

        self.generator = TrainingGenerator()
        self.evaluator = EvaluatorAgent()
        self.reflector = ReflectorAgent()

        from loguru import logger as _logger
        self.logger = _logger

        self.logger.info(
            f"FrameTrainingLoop 初始化 | 循环轮数={iteration_rounds}"
        )

    def process_sample(
        self,
        section_key: str,
        original_content: str,
        research_topic: str,
    ) -> Dict[str, Any]:
        """
        对单个样本执行完整的 G→E→R 训练循环

        Returns:
            {
                "section_key": str,
                "original_content": str,
                "iterations": [
                    {
                        "round": int,
                        "generated_content": str,
                        "evaluation": Dict,
                        "reflection_report": Dict,
                    }, ...
                ],
                "best_iteration": int,       # 最佳轮次索引
                "final_report": Dict,         # 最终的 Reflection Report (存入FAISS用)
            }
        """
        self.logger.info(f"\n{'─'*50}")
        self.logger.info(f"G→E→R 训练循环 | 章节={section_key}")
        self.logger.info(f"{'─'*50}")

        iterations = []
        reflection_history = []

        for round_idx in range(self.iteration_rounds):
            self.logger.info(f"  --- Round {round_idx + 1}/{self.iteration_rounds} ---")

            # Step 1: Generate
            if round_idx == 0:
                gen_content = self.generator.run(
                    section_key=section_key,
                    research_topic=research_topic,
                    reference_context=original_content,
                )
            else:
                gen_content = self.generator.run_with_history(
                    section_key=section_key,
                    original_content=original_content,
                    reflection_reports=reflection_history,
                    iteration_round=round_idx + 1,
                    research_topic=research_topic,
                )

            time.sleep(0.5)

            # Step 2: Evaluate
            evaluation = self.evaluator.run(
                section_key=section_key,
                content=gen_content,
                reference_topic=research_topic,
            )
            time.sleep(0.3)

            # Step 3: Reflect
            reflection_report = self.reflector.run(
                section_key=section_key,
                evaluation_result=evaluation,
            )

            iter_record = {
                "round": round_idx + 1,
                "generated_content": gen_content,
                "evaluation": evaluation,
                "reflection_report": reflection_report,
            }
            iterations.append(iter_record)
            reflection_history.append(reflection_report)

            score = evaluation.get("overall_score", 0)
            self.logger.info(f"  Round {round_idx+1}: score={score:.2f} | "
                           f"gen_len={len(gen_content)}")

            time.sleep(0.5)

        # 选择最佳迭代的报告作为最终输出
        best_idx = max(range(len(iterations)),
                       key=lambda i: iterations[i]["evaluation"].get("overall_score", 0))

        result = {
            "section_key": section_key,
            "research_topic": research_topic,
            "iterations": iterations,
            "best_iteration": best_idx,
            "final_report": iterations[best_idx]["reflection_report"],
            "final_score": iterations[best_idx]["evaluation"].get("overall_score", 0),
        }

        self.logger.info(f"  ✓ 最佳轮次: #{best_idx+1}, 最终分={result['final_score']:.2f}")
        return result


# 导入 time (模块级别使用)
import time