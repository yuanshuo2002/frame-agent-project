"""
Agent 基类 - 所有 Agent 的通用接口
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger


class BaseAgent(ABC):
    """所有 FRAME Agent 的基类"""

    def __init__(self, name: str, client=None):
        self.name = name
        self._client = client
        self.client = client  # 兼容子类直接访问 self.client

        # 统计信息
        self.call_count = 0
        self.total_tokens_used = 0
        self.errors = 0

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """执行 Agent 的核心逻辑（子类必须实现）"""
        pass

    def _record_call(self, success: bool = True):
        """记录调用统计"""
        self.call_count += 1
        if not success:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_calls": self.call_count,
            "errors": self.errors,
            "success_rate": (self.call_count - self.errors) / max(self.call_count, 1),
        }

    def log_info(self, msg: str):
        logger.info(f"[{self.name}] {msg}")

    def log_warning(self, msg: str):
        logger.warning(f"[{self.name}] {msg}")

    def log_error(self, msg: str):
        logger.error(f"[{self.name}] {msg}")
