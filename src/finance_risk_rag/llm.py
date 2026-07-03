"""
Finance-Risk-RAG LLM 客户端模块
==============================
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .config import get_config
from .exceptions import LLMError
from .utils import PIIMasker

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """LLM 客户端封装类"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        enable_masking: bool = True,
    ):
        config = get_config()
        self.api_key = api_key or config.llm_api_key
        self.base_url = base_url or config.llm_base_url
        self.model_name = model_name or config.llm_model_name
        self.enable_masking = enable_masking
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
            raise LLMError(f"Failed to initialize OpenAI client: {e}")

    @property
    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        return self._client is not None

    def _preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """发送前对消息进行脱敏处理"""
        if not self.enable_masking:
            return messages

        processed = []
        for msg in messages:
            processed.append(
                {"role": msg["role"], "content": PIIMasker.mask(msg["content"])}
            )
        return processed

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        """
        发送聊天请求，带有 PII 脱敏和指数退避重试机制。
        """
        if not self.is_available:
            raise LLMError("LLM client not initialized or API key missing.")

        processed_messages = self._preprocess_messages(messages)

        retries = 0
        while retries <= max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=processed_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(
                        f"LLM call failed after {max_retries} retries: {e}"
                    )
                    raise LLMError(f"LLM call failed after {max_retries} retries: {e}")

                wait_time = initial_backoff * (2 ** (retries - 1))
                logger.warning(
                    f"LLM call failed: {e}. Retrying in {wait_time:.2f}s... "
                    f"({retries}/{max_retries})"
                )
                time.sleep(wait_time)

        raise LLMError("Unexpected exit from retry loop.")

    def ask(self, query: str, context: str) -> str:
        """简单的问答接口"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名资深的金融风险分析顾问。"
                    "请根据提供的上下文回答问题，保持专业、严谨且简明。"
                ),
            },
            {
                "role": "user",
                "content": f"参考以下上下文：\n\n{context}\n\n问题：{query}",
            },
        ]
        return self.chat(messages)
