"""
LLM API 统一封装
支持 DeepSeek, OpenAI 及兼容接口
使用 requests 库（避免 httpx 与 vLLM 的 502 兼容性问题）
"""
import os
import json
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    raise ImportError("请先安装 requests: pip install requests")

from loguru import logger
import yaml


# 兼容旧版 openai SDK 的响应结构（保持下游代码不变）
class _ChoiceMessage:
    """模拟 openai choices[0].message 对象"""
    def __init__(self, content: str):
        self.content = content

class _Choice:
    """模拟 openai choices 对象"""
    def __init__(self, content: str):
        self.message = _ChoiceMessage(content)

class _CompletionResponse:
    """模拟 openai chat.completions.create 返回对象"""
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMClient:
    """统一 LLM 客户端"""

    def __init__(self, config: Optional[LLMConfig] = None, config_path: Optional[str] = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_from_yaml(config_path)
        else:
            self.config = LLMConfig()

        # 解析环境变量 (支持 ${VAR} 格式)
        self._resolve_env_vars()

        # 使用 requests Session（避免 httpx 与 vLLM 的 502 兼容性问题）
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        })
        self.api_url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        logger.info(f"初始化 LLM 客户端 (requests): {self.config.provider}/{self.config.model}")

    def _load_from_yaml(self, path: str) -> LLMConfig:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        llm_cfg = cfg.get('llm', {}).get('primary', {})
        return LLMConfig(
            provider=llm_cfg.get('provider', 'deepseek'),
            api_key=llm_cfg.get('api_key', ''),
            base_url=llm_cfg.get('base_url', 'https://api.deepseek.com/v1'),
            model=llm_cfg.get('model', 'deepseek-chat'),
            temperature=float(llm_cfg.get('temperature', 0.7)),
            max_tokens=int(llm_cfg.get('max_tokens', 4096)),
        )

    def _resolve_env_vars(self):
        """解析配置中的环境变量引用"""
        key = self.config.api_key
        if key.startswith('${') and key.endswith('}'):
            env_name = key[2:-1]
            self.config.api_key = os.environ.get(env_name, '')
            if not self.config.api_key:
                logger.warning(f"环境变量 {env_name} 未设置!")

    def _do_request(self, kwargs: Dict) -> str:
        """用 requests 发送请求，返回文本内容"""
        for attempt in range(kwargs.get("retries", 3)):
            try:
                resp = self.session.post(
                    self.api_url,
                    json={
                        "model": self.config.model,
                        "messages": kwargs["messages"],
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    },
                    timeout=120,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return content
            except Exception as e:
                retries = kwargs.get("retries", 3)
                delay = kwargs.get("delay", 1.0)
                logger.warning(f"API 调用失败 (尝试 {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))

        raise RuntimeError(f"API 调用在 {kwargs.get('retries', 3)} 次尝试后仍然失败")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        retries: int = 3,
        delay: float = 1.0,
    ) -> str:
        """
        发送聊天请求，返回文本内容

        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token 数
            response_format: 指定输出格式（requests 模式下暂不使用）
            retries: 重试次数
            delay: 重试间隔(秒)

        Returns:
            模型回复的文本内容
        """
        return self._do_request({
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "retries": retries,
            "delay": delay,
        })

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        发送请求并解析 JSON 响应
        """
        raw = self.chat(
            messages,
            temperature=temperature or 0.3,  # JSON 输出降低温度
            response_format={"type": "json_object"},
            **kwargs,
        )
        # DEBUG: 记录原始响应便于排查解析问题
        logger.debug(f"chat_json raw response ({len(raw)} chars): {raw[:300]}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取 JSON
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            logger.error(f"JSON 解析失败，原始响应:\n{raw[:500]}")
            raise

    def get_evaluator_client(self):
        """获取评估专用客户端 (低温度)"""
        return LLMClient(config=LLMConfig(
            provider=self.config.provider,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            temperature=0.3,
            max_tokens=2048,
        ))


# 全局单例（懒加载）
_global_client: Optional[LLMClient] = None


def get_llm_client(config_path: str = "config/model_config.yaml") -> LLMClient:
    """获取全局 LLM 客户端单例"""
    global _global_client
    if _global_client is None:
        _global_client = LLMClient(config_path=config_path)
    return _global_client


def create_client_from_role(role: str = "primary") -> LLMClient:
    """根据角色创建客户端 (primary / secondary / evaluator)"""
    config_path = "config/model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    role_map = {
        "primary": cfg['llm']['primary'],
        "secondary": cfg['llm']['secondary'],
        "evaluator": cfg['llm'].get('evaluator', cfg['llm']['primary']),
    }

    llm_cfg = role_map.get(role, cfg['llm']['primary'])

    # 解析环境变量
    api_key = llm_cfg.get('api_key', '')
    if api_key.startswith('${') and api_key.endswith('}'):
        env_name = api_key[2:-1]
        api_key = os.environ.get(env_name, '')

    return LLMClient(config=LLMConfig(
        provider=llm_cfg.get('provider', 'deepseek'),
        api_key=api_key,
        base_url=llm_cfg.get('base_url', 'https://api.deepseek.com/v1'),
        model=llm_cfg.get('model', 'deepseek-chat'),
        temperature=float(llm_cfg.get('temperature', 0.7)),
        max_tokens=int(llm_cfg.get('max_tokens', 4096)),
    ))
