"""
Prompt 模板管理 - 论文 Appendix B 的 Prompt 定义
所有 Agent 使用的 Prompt 集中管理
"""

# ============================================================
# Phase 2: 数据集构建 - Extractor Prompts
# ============================================================

EXTRACTOR_PROMPT_TEMPLATE = """
Role: Paper Analyst

Task: Identify and analyze the {section_name} content from the provided article paragraph.

Requirements:
- If the input only contains the paragraph, then extract the {section_name} based on the paragraph.
- If the input includes historical records from previous iterations, it is necessary to reference both
  the paragraph and the previous scores/reasons to provide a better {section_name} extraction.
- Output format: Describe the analysis results in natural language and output in JSON format,
  which should only contain a single key-value pair with the key "{section_key}" and the value being
  the extracted {section_name} description.

Input:
- Current content: The article paragraph (provided below)
- Previous evaluations: Historical records or scores from previous evaluations (if any)

{previous_evaluations}

Article Paragraph:
```
{paragraph}
```

Please output the JSON result:
"""


CHECKER_PROMPT_TEMPLATE = """
Role: {section_name} Evaluator

Task: Evaluate the {section_name} extracted from the provided article paragraph.

Requirements:
- Input format: Two parts - the article paragraph AND the extracted {section_name}
- Output format: Return a JSON object containing exactly two key-value pairs:
  - "score": A numerical score from 0 to 10 (incremented by 0.1) indicating the quality of extraction.
    Scoring rubric:
    * 0-2: Completely irrelevant / poor extraction
    * 2-4: Mostly irrelevant / major information missing
    * 4-6: Partially relevant / significant gaps
    * 6-8: Mostly relevant / minor issues
    * 8-10: Highly relevant / excellent extraction capturing all key points
  - "reason": A textual explanation for the assigned score (2-4 sentences).

Input:
- Article Paragraph:
```
{paragraph}
```
- Extracted {section_name}:
```
{extracted_content}
```

Please evaluate and output the JSON result:
"""


# ============================================================
# Phase 3: FRAME Training Stage - Generator/Evaluator/Reflector
# ============================================================

GENERATOR_PROMPT_TEMPLATE = """
Role: Academic Paper Writer

Task: Based on the provided research topic and reference materials, synthesize a comprehensive,
well-structured {section_name} section for a new medical research paper. Ensure it integrates
key findings while maintaining academic rigor.

Input:
- Research Topic: {research_topic}
- Reference Materials (from related papers): {reference_context}

Requirements for {section_name}:
{section_requirements}

Output a complete, publication-quality {section_name} section in natural language:
"""


EVALUATOR_PROMPT_TEMPLATE = """
Role: Expert Academic Reviewer

Task: Conduct a multi-dimensional quality assessment of the following {section_name} section.

Evaluation Dimensions:
{evaluation_dimensions}

Scoring Scale: 1-5 (increments of 0.1), where:
- 1-2: Poor / Major deficiencies
- 2-3: Below average / Significant gaps
- 3-4: Acceptable / Minor improvements needed
- 4-5: Good / Excellent quality

Content to Evaluate:
```
{content_to_evaluate}
```

Reference Topic (for relevance check): {reference_topic}

Output a JSON object with this exact structure:
{{
    "dimension_scores": {{
        "<dimension_name>": {{"score": <float>, "reason": "<explanation>"}}
        // Include ALL dimensions listed above, each with score and reason
    }},
    "overall_score": <float average of all dimension scores>,
    "summary": "<brief overall assessment>"
}}
"""


REFLECTOR_PROMPT_TEMPLATE = """
Role: Writing Coach & Improvement Specialist

Task: Based on the evaluation feedback, generate a structured reflection report that provides
actionable improvement suggestions for the {section_name} section.

Evaluation Results Received:
{evaluation_results}

Your task is to:
1. Analyze each dimension's score and identify weaknesses (score < 3.5)
2. Map each weakness to specific improvement recommendations
3. Generate actionable suggestions that can guide better generation in future

Improvement Recommendation Mapping:
{recommendation_mapping}

Output a JSON reflection report with this structure:
{{
    "strengths": ["<list of what worked well>"],
    "weaknesses": [
        {{"dimension": "<name>", "score": <float>, "issue": "<description>", "suggestion": "<specific improvement advice>"}}
    ],
    "priority_improvements": ["<top 3 most critical improvements to make>"],
    "reflection_summary": "<2-3 sentence synthesis of key learnings>"
}}
"""


# ============================================================
# Phase 4: Inference - Filter Agent
# ============================================================

FILTER_PROMPT = """
Role: Relevance Gatekeeper

Task: Determine whether the following Reflection Report is truly relevant and useful for generating
a paper on the given research topic.

Research Topic: {research_topic}

Reflection Report (candidate):
```
{report_content}
```

Evaluation Criteria:
1. **Topic Alignment**: Does the report address challenges/methods/findings directly related to the topic?
2. **Actionability**: Can the suggestions in this report actually help improve paper generation?
3. **Specificity**: Are the suggestions concrete enough to be useful?

Output JSON:
{
    "is_relevant": true/false,
    "relevance_score": <0.0-1.0>,
    "reason": "<brief explanation>"
}

Only reports with relevance_score >= 0.6 should be considered relevant.
"""


# ============================================================
# Phase 4: Inference - Integrator
# ============================================================

INTEGRATOR_PROMPT = """
Role: Knowledge Synthesis Specialist

Task: Merge and synthesize multiple reflection reports into a unified, coherent context
that will guide the generation of a high-quality {section_name} section.

Target Research Topic: {research_topic}

Reflection Reports to Integrate:
{reports_block}

Instructions:
- Consolidate overlapping suggestions into unified guidance
- Remove contradictory or redundant information
- Prioritize the most impactful improvement insights
- Produce a balanced, comprehensive synthesis

Output a merged context document in natural language (500-800 words) that captures
the collective wisdom from all reports:
"""


# ============================================================
# 辅助函数: 获取章节对应的 prompt 配置
# ============================================================

SECTION_CONFIG = {
    "topic": {
        "display_name": "Topic / Introduction",
        "section_key": "topic",
        "requirements": """
- Clearly state the core research problem or question
- Establish the significance and motivation of the study
- Provide context for why this problem matters in the medical field
- Be concise but compelling (typically 200-400 words)
""",
    },
    "background": {
        "display_name": "Background",
        "section_key": "background",
        "requirements": """
- Provide essential background knowledge for understanding the research
- Cover relevant prior work and foundational concepts
- Progress from general context to specific research gap
- Include key definitions and establish terminology
- Maintain logical flow toward the research motivation
""",
    },
    "related_work": {
        "display_name": "Related Work",
        "section_key": "related_work",
        "requirements": """
- Systematically review relevant literature across multiple sub-topics
- Critically analyze (not just list) prior approaches and their limitations
- Identify clear research gaps that motivate the current work
- Group related works thematically, not just chronologically
- Connect each reviewed work back to the current research question
""",
    },
    "methodology": {
        "display_name": "Methodology",
        "section_key": "methodology",
        "requirements": """
- Detail the complete experimental design and procedures
- Describe datasets, preprocessing steps, and implementation details
- Explain model architectures, training strategies, and hyperparameters
- Define evaluation metrics and statistical methods
- Provide enough detail for reproducibility
""",
    },
    "result": {
        "display_name": "Result",
        "section_key": "result",
        "requirements": """
- Present quantitative results with precise numerical values
- Include comparative analysis against baselines
- Report statistical significance measures where applicable
- Use tables and figures references appropriately
- Address each research question with corresponding evidence
""",
    },
    "conclusion": {
        "display_name": "Conclusion",
        "section_key": "conclusion",
        "requirements": """
- Summarize key contributions and main findings concisely
- Discuss practical implications for the medical field
- Acknowledge limitations honestly
- Propose meaningful future research directions
- End with a strong take-away message
""",
    },
}


def get_extractor_prompt(section_key: str, paragraph: str, previous_evaluations: str = "") -> str:
    """获取 Extractor 的完整 prompt"""
    config = SECTION_CONFIG.get(section_key, SECTION_CONFIG["background"])
    return EXTRACTOR_PROMPT_TEMPLATE.format(
        section_name=config["display_name"],
        section_key=config["section_key"],
        previous_evaluations=previous_evaluations or "No previous evaluations available.",
        paragraph=paragraph,
    )


def get_checker_prompt(section_key: str, paragraph: str, extracted_content: str) -> str:
    """获取 Checker 的完整 prompt"""
    config = SECTION_CONFIG.get(section_key, SECTION_CONFIG["background"])
    return CHECKER_PROMPT_TEMPLATE.format(
        section_name=config["display_name"],
        paragraph=paragraph,
        extracted_content=extracted_content,
    )


def get_generator_prompt(section_key: str, research_topic: str, reference_context: str) -> str:
    """获取 Generator 的完整 prompt"""
    config = SECTION_CONFIG.get(section_key, SECTION_CONFIG["background"])
    return GENERATOR_PROMPT_TEMPLATE.format(
        section_name=config["display_name"],
        research_topic=research_topic,
        reference_context=reference_context or "No reference materials available.",
        section_requirements=config["requirements"],
    )


def get_evaluator_prompt(section_key: str, content: str, reference_topic: str, dimensions_info: str) -> str:
    """获取 Evaluator 的完整 prompt"""
    config = SECTION_CONFIG.get(section_key, SECTION_CONFIG["background"])
    return EVALUATOR_PROMPT_TEMPLATE.format(
        section_name=config["display_name"],
        evaluation_dimensions=dimensions_info,
        content_to_evaluate=content,
        reference_topic=reference_topic,
    )


def get_reflector_prompt(section_key: str, eval_result: str, rec_mapping: str) -> str:
    """获取 Reflector 的完整 prompt"""
    config = SECTION_CONFIG.get(section_key, SECTION_CONFIG["background"])
    return REFLECTOR_PROMPT_TEMPLATE.format(
        section_name=config["display_name"],
        evaluation_results=eval_result,
        recommendation_mapping=rec_mapping,
    )
