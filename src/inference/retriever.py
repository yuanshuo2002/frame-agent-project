"""
FAISS 向量数据库管理 - RAG 检索的核心存储层
对应论文 5.1: We employed FAISS as our database software
"""
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss 未安装! 请执行: pip install faiss-cpu")


class FAISSVectorStore:
    """
    FAISS 向量数据库管理器

    功能:
    1. 存储 Reflection Report 的向量表示
    2. 基于相似度的 Top-K 检索 (RAG)
    3. 支持增量添加和持久化
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "IndexFlatIP",
        index_dir: str = "data/faiss_index/",
        embedding_model=None,
    ):
        if not FAISS_AVAILABLE:
            raise RuntimeError("请先安装 faiss-cpu: pip install faiss-cpu")

        self.dimension = dimension
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.index = None
        self.metadata: List[Dict[str, Any]] = []  # 存储每条记录的元数据

        os.makedirs(index_dir, exist_ok=True)

        # 初始化索引
        if index_type == "IndexFlatIP":
            # 内积索引（配合归一化向量 = 余弦相似度）
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexIVFFlat":
            # IVF 索引（大规模数据时更高效，需要先训练）
            nlist = min(4096, max(4, int(np.sqrt(10000))))  # 聚类中心数
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        logger.info(f"FAISS 索引初始化 | 类型={index_type} | 维度={dimension}")

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_reports(self, reports: List[Dict[str, Any]], text_field: str = "raw_text") -> int:
        """
        批量添加 Reflection Report 到向量库

        Args:
            reports: Reflection Report 列表
            text_field: 用于 Embedding 的文本字段名

        Returns:
            添加的向量数量
        """
        if not reports:
            logger.warning("没有可添加的报告")
            return 0

        # 过滤掉空文本
        valid_reports = []
        texts_to_embed = []
        for r in reports:
            text = r.get(text_field, "") or r.get("reflection_summary", "")
            if len(text) < 10:
                continue
            valid_reports.append(r)
            texts_to_embed.append(text)

        if not valid_reports or not texts_to_embed:
            logger.warning("所有报告文本都为空")
            return 0

        # 生成 embeddings
        embeddings = self.embedding_model.encode(
            texts_to_embed,
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings).astype('float32')

        # 添加到索引
        start_id = self.total_vectors
        self.index.add(embeddings)

        # 记录元数据
        for i, report in enumerate(valid_reports):
            meta = {
                "vector_id": start_id + i,
                "section_key": report.get("section_key", "unknown"),
                "research_topic": report.get("research_topic", ""),
                "reflection_summary": report.get("reflection_summary", ""),
                "source_eval_score": report.get("source_eval_score", 0),
                "priority_improvements": report.get("priority_improvements", []),
                "raw_text_len": len(texts_to_embed[i]),
                **report,  # 包含原始全部信息
            }
            self.metadata.append(meta)

        added = len(valid_reports)
        logger.info(f"✓ 已添加 {added} 个向量到 FAISS | 总计={self.total_vectors}")
        return added

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        相似度搜索

        Args:
            query_text: 查询文本
            top_k: 返回 Top-K 结果
            section_filter: 可选，按章节类型过滤

        Returns:
            (结果元数据列表, 相似度分数数组)
        """
        if self.total_vectors == 0:
            logger.warning("FAISS 索引为空!")
            return [], np.array([])

        # 对查询文本编码
        query_vec = self.embedding_model.encode([query_text])
        query_vec = np.array(query_vec).astype('float32')

        # 扩大搜索范围（因为后续可能需要过滤）
        search_k = top_k * 3 if section_filter else top_k
        scores, indices = self.index.search(query_vec, min(search_k, self.total_vectors))

        results = []
        valid_scores = []
        seen_ids = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            vec_id = meta.get("vector_id")

            # 去重
            if vec_id in seen_ids:
                continue
            seen_ids.add(vec_id)

            # 章节过滤
            if section_filter and meta.get("section_key") != section_filter:
                continue

            result_meta = {**meta, "similarity_score": float(score)}
            results.append(result_meta)
            valid_scores.append(score)

            if len(results) >= top_k:
                break

        return results, np.array(valid_scores) if valid_scores else np.array([])

    def save(self, name: str = "frame_index"):
        """持久化到磁盘"""
        idx_path = os.path.join(self.index_dir, f"{name}.index")
        meta_path = os.path.join(self.index_dir, f"{name}.metadata.pkl")

        try:
            # 对于 IVF 索引需要特殊处理
            if hasattr(self.index, 'ntotal'):
                faiss.write_index(self.index, idx_path)

            with open(meta_path, 'wb') as f:
                pickle.dump({
                    "metadata": self.metadata,
                    "dimension": self.dimension,
                    "total": self.total_vectors,
                }, f)

            logger.info(f"💾 FAISS 索引已保存 | 路径={idx_path} | 向量数={self.total_vectors}")
        except Exception as e:
            logger.error(f"保存失败: {e}")

    def load(self, name: str = "frame_index") -> bool:
        """从磁盘加载"""
        idx_path = os.path.join(self.index_dir, f"{name}.index")
        meta_path = os.path.join(self.index_dir, f"{name}.metadata.pkl")

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            logger.warning(f"索引文件不存在: {idx_path}")
            return False

        try:
            self.index = faiss.read_index(idx_path)
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            logger.info(f"📂 FAISS 索引已加载 | 向量数={self.total_vectors} | 维度={self.dimension}")
            return True
        except Exception as e:
            logger.error(f"加载失败: {e}")
            return False


def build_faiss_from_training_results(
    training_results: List[Dict[str, Any]],
    embedding_model,
    output_dir: str = "data/faiss_index/",
    dimension: int = 1024,
) -> FAISSVectorStore:
    """
    从训练结果批量构建 FAISS 索引

    Args:
        training_results: FrameTrainingLoop.process_sample 的输出列表
        embedding_model: EmbeddingModel 实例
        output_dir: 输出目录

    Returns:
        构建好的 FAISSVectorStore 实例
    """
    store = FAISSVectorStore(
        dimension=dimension,
        index_dir=output_dir,
        embedding_model=embedding_model,
    )

    # 收集所有 final_report
    all_reports = []
    for sample_result in training_results:
        final_report = sample_result.get("final_report")
        if final_report:
            final_report["research_topic"] = sample_result.get("research_topic", "")
            all_reports.append(final_report)

    store.add_reports(all_reports)
    store.save()
    return store
