"""
LLM client wrapper for the Finance-Risk-RAG system.
"""

import logging
from typing import Dict, List, Optional

from openai import OpenAI

from .config import get_config
from .exceptions import LLMError


class LLMClientWrapper:
    """Wrapper for OpenAI-compatible LLM clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        config = get_config()
        self.api_key = api_key or config.llm_api_key
        self.base_url = base_url or config.llm_base_url
        self.model_name = model_name or config.llm_model_name
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            self.logger.warning("No LLM API key provided.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                raise LLMError(f"Failed to initialize LLM client: {e}")

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> str:
        """Send a chat completion request."""
        if not self.client:
            raise LLMError("LLM client not initialized.")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise LLMError(f"LLM call failed: {e}")

    def ask(self, query: str, context: str) -> str:
        """Simplified ask interface with context."""
        messages = [
            {"role": "system", "content": "You are a professional financial risk analyst. Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        return self.chat(messages)
