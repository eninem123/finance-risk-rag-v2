"""
Finance-Risk-RAG LLM 客户端
===========================

封装 LLM 调用逻辑，支持 OpenAI 兼容接口。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Protocol

from .config import LLM_API_KEY, LLM_BASE_URL

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """LLM客户端协议"""
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...


class LLMClientWrapper:
    """LLM客户端封装类"""

    DEFAULT_MODEL = "moonshot-v1-8k"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 512

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = DEFAULT_MODEL
    ) -> None:
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称

        Raises:
            Exception: 客户端初始化失败
        """
        self._api_key = api_key or LLM_API_KEY
        self._base_url = base_url or LLM_BASE_URL
        self._model_name = model_name
        self._client: Optional[Any] = None

        if not self._api_key:
            logger.warning(
                "未检测到 LLM API key。请设置环境变量 OPENAI_API_KEY 或 MOONSHOT_API_KEY。"
            )
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            logger.info(f"LLM客户端初始化成功，模型: {self._model_name}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise e

    @property
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self._client is not None

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> str:
        """
        通用 LLM 调用

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLM回答
        """
        if not self.is_available:
            raise RuntimeError("LLM客户端未初始化，请设置API密钥")

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise e

    def ask(
        self,
        query: str,
        context: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """
        参考上下文向LLM提问
        """
        system_prompt = "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}"}
        ]
        return self.call(messages, temperature, max_tokens)
