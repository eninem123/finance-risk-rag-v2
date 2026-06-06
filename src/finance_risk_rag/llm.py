"""
Finance-Risk-RAG LLM 客户端模块
==============================
"""

import logging
from typing import Any, Dict, List, Optional

from .config import get_config
from .exceptions import LLMError

logger = logging.getLogger(__name__)

class LLMClientWrapper:
    """LLM 客户端封装类"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
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

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1000) -> str:
        if not self.is_available:
            raise LLMError("LLM client not initialized.")
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")

    def ask(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。"},
            {"role": "user", "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}"}
        ]
        return self.chat(messages)
