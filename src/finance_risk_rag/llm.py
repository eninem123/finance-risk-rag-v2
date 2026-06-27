"""
Finance-Risk-RAG LLM 客户端模块
==============================
"""

import logging
import time
from typing import Dict, List, Optional

from .config import get_config
from .exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """LLM 客户端封装类"""

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
        self._client = None

        if not self.api_key:
            logger.warning("LLM API key not found. LLM features will be disabled.")
            return

        self._initialize_client()

    def _initialize_client(self):
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
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
        """
        if not self.is_available:
            raise LLMError("LLM client not initialized or API key missing.")

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
                    err_msg = f"LLM call failed after {max_retries} retries: {str(e)}"
                    logger.error(err_msg)
                    raise LLMError(err_msg)

                wait_time = initial_backoff * (2 ** (retries - 1))
                logger.warning(
                    f"LLM call failed: {e}. Retrying in {wait_time:.2f}s... "
                    f"({retries}/{max_retries})"
                )
                time.sleep(wait_time)

        raise LLMError("Unexpected exit from retry loop.")

    def ask(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名资深的金融风险分析专家。请根据提供的上下文，"
                    "以专业、严谨且条理清晰的方式回答用户问题。如果上下文中没有信息，"
                    "请诚实说明。"
                ),
            },
            {
                "role": "user",
                "content": f"【参考上下文】\n{context}\n\n【用户问题】\n{query}",
            },
        ]
        return self.chat(messages)
