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
            logger.warning("LLM API key not found.")
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

        # Should not reach here
        raise LLMError("Unexpected exit from retry loop.")

    def ask(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名资深的金融风险分析专家。在回答问题时，请遵循以下原则：\n"
                    "1. **逻辑推理**：采用思维链（Chain-of-Thought）方式，先分析上下文中的关键风险因素，再得出结论。\n"
                    "2. **证据引用**：必须引用上下文中的具体数据或陈述来支持你的分析。\n"
                    "3. **多维评估**：从财务、运营、市场和法律等多个维度评估潜在风险。\n"
                    "4. **专业性**：使用专业的金融风险管理术语。"
                ),
            },
            {
                "role": "user",
                "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}",
            },
        ]
        return self.chat(messages)
