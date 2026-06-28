"""
Finance-Risk-RAG LLM 客户端模块
==============================

提供统一的 LLM 访问接口，包含自动重试、异常处理和 prompt 管理。
"""

import logging
import time
from typing import Dict, List, Optional

from .config import get_config
from .exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """
    LLM 客户端封装类，支持 OpenAI 兼容接口。

    Attributes:
        api_key (str): API 访问密钥。
        base_url (str): API 基础 URL。
        model_name (str): 使用的模型名称。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        初始化 LLM 客户端。

        Args:
            api_key: API 密钥，若为 None 则从配置读取。
            base_url: 基础 URL，若为 None 则从配置读取。
            model_name: 模型名称，若为 None 则从配置读取。
        """
        config = get_config()
        self.api_key = api_key or config.llm_api_key
        self.base_url = base_url or config.llm_base_url
        self.model_name = model_name or config.llm_model_name
        self._client = None

        if not self.api_key:
            logger.warning("LLM API key not found.")
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """初始化 OpenAI 客户端实例。"""
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}")

    @property
    def is_available(self) -> bool:
        """检查 LLM 客户端是否可用。"""
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        """
        发送聊天请求，带有指数退避重试机制。

        Args:
            messages: 消息列表。
            temperature: 生成随机度。
            max_tokens: 最大生成长度。
            max_retries: 最大重试次数。
            initial_backoff: 初始等待时间（秒）。

        Returns:
            str: 模型返回的文本。

        Raises:
            LLMError: 调用失败时抛出。
        """
        if not self.is_available:
            raise LLMError("LLM client not initialized.")

        retries = 0
        while retries <= max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"LLM call failed after {max_retries} retries: {e}")
                    raise LLMError(f"LLM call failed after {max_retries} retries: {e}")

                wait_time = initial_backoff * (2 ** (retries - 1))
                logger.warning(
                    f"LLM call failed: {e}. Retrying in {wait_time:.2f}s... "
                    f"({retries}/{max_retries})"
                )
                time.sleep(wait_time)

        raise LLMError("Unexpected exit from retry loop.")

    def ask(self, query: str, context: str) -> str:
        """
        基于上下文回答问题。

        Args:
            query: 用户问题。
            context: 检索到的相关上下文。

        Returns:
            str: AI 回答。
        """
        messages = [
            {
                "role": "system",
                "content": "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。",
            },
            {
                "role": "user",
                "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}",
            },
        ]
        return self.chat(messages)
