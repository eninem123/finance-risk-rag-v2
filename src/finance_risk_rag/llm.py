"""
Finance-Risk-RAG LLM 客户端模块
==============================
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from .config import get_config
from .exceptions import LLMError
from .utils import PIIMasker

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """LLM 客户端封装类，支持同步和异步调用，集成 PII 脱敏功能"""

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
        self._masker = PIIMasker() if enable_masking else None

        self._client = None
        self._async_client = None

        if not self.api_key:
            logger.warning("LLM API key not found.")
            return

        self._initialize_clients()

    def _initialize_clients(self):
        try:
            from openai import AsyncOpenAI, OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI clients: {e}")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def _mask_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """如果启用了脱敏，则对消息内容进行脱敏"""
        if not self.enable_masking or not self._masker:
            return messages

        masked_messages = []
        for msg in messages:
            masked_msg = msg.copy()
            if msg.get("role") == "user":
                masked_msg["content"] = self._masker.mask(msg["content"])
            masked_messages.append(masked_msg)
        return masked_messages

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        """
        同步发送聊天请求，带有指数退避重试机制。
        """
        if not self.is_available:
            raise LLMError("LLM client not initialized.")

        messages = self._mask_messages(messages)

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

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        """
        异步发送聊天请求，带有指数退避重试机制。
        """
        if not self._async_client:
            raise LLMError("Async LLM client not initialized.")

        messages = self._mask_messages(messages)

        retries = 0
        while retries <= max_retries:
            try:
                response = await self._async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Async LLM call failed after {max_retries} retries: {e}")
                    raise LLMError(f"Async LLM call failed after {max_retries} retries: {e}")

                wait_time = initial_backoff * (2 ** (retries - 1))
                logger.warning(
                    f"Async LLM call failed: {e}. Retrying in {wait_time:.2f}s... "
                    f"({retries}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

        raise LLMError("Unexpected exit from retry loop.")

    def ask(self, query: str, context: str) -> str:
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

    async def aask(self, query: str, context: str) -> str:
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
        return await self.achat(messages)
