"""
Embedding 封装 - 支持本地模型和 OpenAI API
用于 FAISS 向量检索
"""
import os
import numpy as np
from typing import List, Optional, Union
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingModel:
    """Embedding 模型封装"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = None,  # 改为可选，自动检测
        use_openai: bool = False,
        openai_api_key: str = "",
    ):
        self.model_name = model_name
        # 自动检测设备: 有 CUDA 用 GPU，否则自动降级 CPU
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
                logger.info(f"检测到 GPU: {gpu_name} ({vram_gb:.1f} GB)")
            else:
                self.device = "cpu"
                logger.info("未检测到 CUDA GPU，使用 CPU 模式 (Embedding 会较慢但可用)")
        else:
            self.device = device
        self.use_openai = use_openai

        if self.use_openai:
            self._init_openai(openai_api_key)
        else:
            self._init_local()

    def _init_local(self, max_retries: int = 3):
        """初始化本地 Sentence Transformer 模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        for attempt in range(max_retries):
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"加载 Embedding 模型: {self.model_name}, 维度={self.dimension}, 设备={self.device}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"模型加载失败 (尝试 {attempt+1}): {e}")
                    import time; time.sleep(3)
                else:
                    raise

    def _init_openai(self, api_key: str):
        """初始化 DashScope/OpenAI 兼容的 Embedding API"""
        import os
        from openai import OpenAI

        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model_name = os.environ.get("EMBEDDING_MODEL", "text-embedding-v1")

        self.openai_client = OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=base_url,
        )
        # 百炼 text-embedding-v1 维度=1536 (与 text-embedding-ada-002 一致)
        self.dimension = int(os.environ.get("EMBEDDING_DIM", "1536"))
        self.embedding_api_model = model_name
        logger.info(f"使用 API Embedding | 模型={model_name} | 维度={self.dimension}")

    @property
    def dim(self) -> int:
        return self.dimension

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            texts: 单个字符串或字符串列表
            batch_size: 批处理大小 (仅本地模型)
            show_progress_bar: 是否显示进度条

        Returns:
            形状为 (n_texts, dimension) 的 numpy 数组
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.use_openai:
            return self._encode_openai(texts)
        else:
            return self._encode_local(texts, batch_size, show_progress_bar)

    def _encode_local(self, texts: List[str], batch_size: int, show_progress_bar: bool) -> np.ndarray:
        """使用本地模型编码"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,  # L2 归一化，便于内积计算相似度
        )
        return np.array(embeddings).astype('float32')

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """使用 DashScope/OpenAI API 编码"""
        all_embeddings = []
        # 使用配置中的模型名（默认 text-embedding-v1）
        model = getattr(self, 'embedding_api_model', 'text-embedding-v1')
        # DashScope text-embedding API: max 25 per batch
        max_batch = 25
        for i in range(0, len(texts), max_batch):
            batch = texts[i:i+max_batch]
            response = self.openai_client.embeddings.create(
                model=model,
                input=batch,
            )
            batch_emb = [item.embedding for item in response.data]
            all_embeddings.extend(batch_emb)

        result = np.array(all_embeddings).astype('float32')
        # L2 归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-10)
        return result


# 全局单例
_global_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    config_path: str = "config/model_config.yaml",
) -> EmbeddingModel:
    """获取全局 Embedding 模型单例"""
    global _global_model
    if _global_model is None:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        emb_cfg = cfg.get('embedding', {})

        _global_model = EmbeddingModel(
            model_name=emb_cfg.get('model_name', 'text-embedding-v1'),
            device=emb_cfg.get('device', None),
            use_openai=emb_cfg.get('provider') == 'openai',
            openai_api_key=os.environ.get('DASHSCOPE_API_KEY', ''),
        )
    return _global_model
