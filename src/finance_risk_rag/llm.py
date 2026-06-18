import logging
import time
from typing import Dict, List, Optional

from openai import OpenAI

from .exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """封装 LLM 客户端，支持重试和统一接口"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-3.5-turbo",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        if not self.is_available:
            raise LLMError("LLM client not initialized (missing API key)")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM chat failed after {max_retries} attempts: {e}")
                    raise LLMError(f"LLM call failed after {max_retries} retries: {e}")
                time.sleep(initial_backoff * (2**attempt))
        return ""

    def ask(self, question: str, context: str) -> str:
        """针对 RAG 优化的提问接口"""
        prompt = f"""
你是一个专业的金融风险控制专家。请根据以下提供的参考文本回答用户的问题。
如果参考文本中没有相关信息，请诚实告知。

【参考文本】
{context}

【用户问题】
{question}

请以专业、客观的角度进行回答，并指出风险点（如有）。
"""
        messages = [{"role": "user", "content": prompt}]
        # Use implicit string concatenation to avoid E501
        logger.info(
            f"Sending RAG query to LLM (model: {self.model_name}, "
            f"context length: {len(context)})"
        )
        return self.chat(messages, temperature=0.3)
