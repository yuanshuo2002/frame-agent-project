"""
端到端推理流水线 - 对应论文 Section 4.2 Inference Stage
完整流程: Research Topic → RAG检索 → Filter筛选 → Integrator合并 → Generator生成
"""
import os
import sys
import time
import json
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger
import yaml

# 导入各组件
from src.inference.retriever import FAISSVectorStore, build_faiss_from_training_results
from src.inference.filter_agent import FilterAgent
from src.inference.integrator import IntegratorAgent
from src.agents.generator import GeneratorAgent


class FrameInferencePipeline:
    """
    FRAME 推理流水线 - 完整的端到端论文生成流程

    对应论文 Figure 1 下半部分 (Inference Stage):
        Research Topic → RAG → Filter → Integrator → Generator → New Paper
    """

    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        faiss_store: Optional[FAISSVectorStore] = None,
        embedding_model=None,
    ):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        inf_cfg = self.config.get('inference', {})
        faiss_cfg = self.config.get('faiss', {})

        # 初始化各组件
        self.faiss_store = faiss_store  # 外部传入或后续加载
        self.embedding_model = embedding_model
        self.generator = GeneratorAgent()
        self.filter_agent = FilterAgent() if inf_cfg.get('filter_enabled', True) else None
        self.integrator = IntegratorAgent() if inf_cfg.get('integrator_enabled', True) else None

        self.retrieval_top_k = int(faiss_cfg.get('retrieval_top_k', 5))
        self.num_to_integrate = int(inf_cfg.get('num_reports_to_integrate', 3))

        logger.info(
            f"FrameInferencePipeline 初始化完成 | "
            f"TopK={self.retrieval_top_k} | Filter={'✓' if self.filter_agent else '✗'} | "
            f"Integrator={'✓' if self.integrator else '✗'}"
        )

    def set_faiss_store(self, store: FAISSVectorStore):
        """设置/替换 FAISS 向量库"""
        self.faiss_store = store
        self.embedding_model = store.embedding_model
        logger.info(f"FAISS Store 已绑定 | 总向量数={store.total_vectors}")

    def generate_section(
        self,
        research_topic: str,
        section_key: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        为指定章节执行完整的推理流程

        Args:
            research_topic: 研究主题
            section_key: 目标章节 (topic/background/related_work/methodology/result/conclusion)
            **kwargs: 额外参数

        Returns:
            {
                "section_key": str,
                "research_topic": str,
                "generated_content": str,
                "retrieved_count": int,
                "filtered_count": int,
                "integrated_context": str,
                "method": str,  # "no_rag" / "rag" / "ours" (filter+integrate)
            }
        """
        method = kwargs.pop("method", "ours")  # 默认使用完整 FRAME 流程
        use_filter = (method == "ours") and (self.filter_agent is not None)
        use_integrate = (method in ["rag", "ours"]) and (self.integrator is not None)

        logger.info(f"\n{'='*50}")
        logger.info(f"📝 生成 [{section_key}] | Topic: {research_topic[:60]}")
        logger.info(f"   方法={method} | Filter={'✓' if use_filter else '✗'} | Integrate={'✓' if use_integrate else '✗'}")

        context = ""
        retrieved_count = 0
        filtered_count = 0

        # Step 1: RAG 检索（除非是 no_rag 模式）
        if method != "no_rag" and self.faiss_store and self.faiss_store.total_vectors > 0:
            candidates, scores = self.faiss_store.search(
                query_text=research_topic,
                top_k=self.retrieval_top_k,
                section_filter=section_key,  # 可选：按章节过滤
            )
            retrieved_count = len(candidates)

            if candidates:
                # Step 2: Filter 筛选
                if use_filter:
                    filtered = self.filter_agent.batch_filter(
                        research_topic=research_topic,
                        candidates=candidates,
                        min_keep=1,
                    )
                    filtered_count = len(filtered)
                else:
                    filtered = candidates
                    filtered_count = len(filtered)

                # Step 3: Integrator 合并
                if use_integrate and filtered:
                    context = self.integrator.integrate(
                        section_key=section_key,
                        research_topic=research_topic,
                        filtered_reports=filtered,
                    )
                elif filtered:
                    # 无 Integrator，直接用第一份报告
                    r = filtered[0]
                    context = r.get("reflection_summary", r.get("raw_text", ""))

        # Step 4: Generator 生成最终内容
        generated = self.generator.run(
            section_key=section_key,
            research_topic=research_topic,
            reference_context=context,
            **kwargs,
        )

        result = {
            "section_key": section_key,
            "research_topic": research_topic,
            "generated_content": generated,
            "retrieved_count": retrieved_count,
            "filtered_count": filtered_count,
            "integrated_context": context[:500] if context else "",
            "method": method,
            "content_length": len(generated),
        }

        logger.info(f"✅ [{section_key}] 生成完成 | 方法={method} | "
                   f"检索={retrieved_count} → 过滤={filtered_count} | 输出长度={len(generated)}")

        return result

    def generate_full_paper(
        self,
        research_topic: str,
        sections: Optional[List[str]] = None,
        method: str = "ours",
    ) -> Dict[str, Any]:
        """
        生成一篇完整的医学研究论文的所有章节

        Args:
            research_topic: 研究主题
            sections: 要生成的章节列表（默认全部6个）
            method: "no_rag" | "rag" | "ours"

        Returns:
            包含所有章节结果的字典
        """
        all_sections = [
            "topic", "background", "related_work",
            "methodology", "result", "conclusion",
        ]
        target_sections = sections or all_sections

        paper_result = {
            "research_topic": research_topic,
            "method": method,
            "sections": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        for sec_key in target_sections:
            try:
                result = self.generate_section(
                    research_topic=research_topic,
                    section_key=sec_key,
                    method=method,
                )
                paper_result["sections"][sec_key] = result
                time.sleep(1)  # API 节流
            except Exception as e:
                logger.error(f"生成章节 [{sec_key}] 失败: {e}")
                paper_result["sections"][sec_key] = {
                    "error": str(e), "section_key": sec_key,
                }

        return paper_result

    def run_comparison_experiment(
        self,
        research_topics: List[str],
        sections: List[str] = None,
    ) -> Dict[str, List[Dict]]:
        """
        运行三组对比实验: No-RAG vs RAG vs Ours(Filter)
        
        这是论文核心实验 (Section 5.3) 的复现

        Returns:
            {"no_rag": [...], "rag": [...], "ours": [...]}
        """
        methods = ["no_rag", "rag", "ours"]
        results = {m: [] for m in methods}

        total = len(research_topics)
        for i, topic in enumerate(research_topics):
            logger.info(f"\n{'#'*50}")
            logger.info(f"对比实验进度: {i+1}/{total} | {topic[:60]}...")
            logger.info(f"{'#'*50}")

            for method in methods:
                try:
                    paper = self.generate_full_paper(topic, sections=sections, method=method)
                    results[method].append(paper)
                    logger.info(f"  ✓ 方法 [{method}] 完成")
                    time.sleep(2)  # 跨方法节流
                except Exception as e:
                    logger.error(f"  ✗ 方法 [{method}] 失败: {e}")

        # 统计摘要
        for m in methods:
            success = sum(1 for p in results[m] if "error" not in str(p))
            logger.info(f"\n结果统计 [{m}]: {success}/{len(results[m])} 成功")

        return results


def load_or_build_faiss_index(
    training_results_path: str = "checkpoints/training_results.json",
    index_name: str = "frame_index",
    config_path: str = "config/model_config.yaml",
) -> FAISSVectorStore:
    """
    加载已有的 FAISS 索引或从训练数据重新构建

    Args:
        training_results_path: 训练结果 JSON 路径（FrameTrainingLoop 输出）
        index_name: 索引名称
        config_path: 配置文件路径

    Returns:
        FAISSVectorStore 实例
    """
    from src.utils.embedding import get_embedding_model

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    emb_cfg = cfg.get('embedding', {})
    faiss_cfg = cfg.get('faiss', {})

    # 初始化 Embedding 模型（获取实际维度）
    emb_model = get_embedding_model(config_path)
    actual_dim = emb_model.dimension

    store = FAISSVectorStore(
        dimension=actual_dim,
        index_dir=str(faiss_cfg.get('index_dir', 'data/faiss_index/')),
        embedding_model=emb_model,
    )

    # 尝试加载已有索引（检查维度是否匹配）
    if store.load(index_name):
        if store.dimension != actual_dim or store.index.d != actual_dim:
            logger.warning(
                f"FAISS 索引维度不匹配! 索引={store.index.d} vs 模型={actual_dim} → 重建索引"
            )
            store = FAISSVectorStore(
                dimension=actual_dim,
                index_dir=str(faiss_cfg.get('index_dir', 'data/faiss_index/')),
                embedding_model=emb_model,
            )
        else:
            return store

    # 从训练数据构建索引
    # 支持两种数据格式：
    #   格式1 (FrameTrainingLoop输出): {paper_id: {paper_title, section_results: {topic:{...},...}}}
    #   格式2 (列表): [{paper_title, sections: {topic:{content,...},...}}, ...]
    train_data = None
    for path in [training_results_path, "checkpoints/training_results.json"]:
        if os.path.exists(path):
            logger.info(f"从训练数据构建 FAISS 索引: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            break

    if not train_data:
        logger.warning("未找到训练数据，返回空 FAISS Store")
        return store

    all_reports = []

    if isinstance(train_data, dict):
        # 格式1: {paper_id: {paper_title, section_results: {...}}}
        for paper_id, sample in train_data.items():
            title = sample.get("paper_title", "")
            sec_results = sample.get("section_results", {})
            for sec_key, sec_data in sec_results.items():
                # 优先用 final_report.generated_content 或 final_report.raw_text
                content = ""
                fr = sec_data.get("final_report", {})
                if isinstance(fr, dict):
                    content = fr.get("raw_text", "") or fr.get("generated_content", "")
                # fallback: iterations[0].generated_content
                if not content:
                    iters = sec_data.get("iterations", [])
                    if iters and isinstance(iters[0], dict):
                        content = iters[0].get("generated_content", "")

                if len(content) > 20:
                    all_reports.append({
                        "section_key": sec_key,
                        "research_topic": title,
                        "reflection_summary": fr.get("reflection_summary", "") if isinstance(fr, dict) else "",
                        "raw_text": content,
                        "source_eval_score": float(sec_data.get("final_score", 5.0)),
                        "priority_improvements": fr.get("priority_improvements", []) if isinstance(fr, dict) else [],
                    })

    elif isinstance(train_data, list):
        # 格式2: [{paper_title, sections: {...}}]
        for sample in train_data:
            sections = sample.get("sections") or sample.get("section_results", {})
            title = sample.get("paper_title", "")
            for sec_key, sec_data in sections.items():
                content = ""
                if isinstance(sec_data, dict):
                    content = sec_data.get("content", "") or sec_data.get("raw_text", "")
                    if not content:
                        fr = sec_data.get("final_report", {})
                        if isinstance(fr, dict):
                            content = fr.get("raw_text", "") or fr.get("generated_content", "")
                if len(content) > 20:
                    all_reports.append({
                        "section_key": sec_key,
                        "research_topic": title,
                        "raw_text": content,
                        "source_eval_score": 5.0,
                        "priority_improvements": [],
                    })

    if all_reports:
        store.add_reports(all_reports)
        store.save(index_name)
        logger.info(f"FAISS 索引构建完成 | 总向量={store.total_vectors} | 维度={actual_dim}")
    else:
        logger.warning("训练结果中没有可用的文本内容")

    return store
