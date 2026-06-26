"""
Finance-Risk-RAG LLM 客户端模块
==============================

提供统一的 LLM 访问接口，包含自动重试、异常处理和多种模型支持。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import get_config
from .exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """LLM 客户端封装类，支持金融级的健壮性要求。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        config = get_config()
        self.api_key = api_key or config.llm_api_key
        self.base_url = base_url or config.llm_base_url
        self.model_name = model_name or config.llm_model_name
        self._client: Any = None

        if not self.api_key:
            logger.warning("LLM API key not found. LLM features will be disabled.")
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """初始化 OpenAI 兼容客户端"""
        try:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"LLM client initialized with model: {self.model_name}")
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}")

    @property
    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1500,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        """
        发送聊天请求，带有指数退避重试机制。

        Args:
            messages: 消息列表
            temperature: 生成温度
            max_tokens: 最大生成长度
            max_retries: 最大重试次数
            initial_backoff: 初始重试延迟（秒）

        Returns:
            str: LLM 返回的文本内容
        """
        if not self.is_available:
            raise LLMError("LLM client not initialized. Check your API key.")

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
                    err_msg = f"LLM call failed after {max_retries} retries. Error: {str(e)}"
                    logger.error(err_msg)
                    raise LLMError(err_msg) from e

                wait_time = initial_backoff * (2 ** (retries - 1))
                logger.warning(
                    f"LLM call encountered error: {e}. "
                    f"Retrying in {wait_time:.2f}s... ({retries}/{max_retries})"
                )
                time.sleep(wait_time)

        raise LLMError("Unexpected termination of LLM retry loop.")

    def ask(self, query: str, context: str) -> str:
        """
        基于上下文回答财务风险相关问题。
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名资深的金融风控合规专家。你的任务是根据提供的文档上下文，"
                    "准确、客观地回答财务风险相关问题。如果上下文中没有信息，请明确告知。"
                    "请使用专业、简洁的语言。"
                ),
            },
            {
                "role": "user",
                "content": f"【参考上下文】\n{context}\n\n【风险提问】\n{query}\n\n【专家回答】",
            },
        ]
        return self.chat(messages)
