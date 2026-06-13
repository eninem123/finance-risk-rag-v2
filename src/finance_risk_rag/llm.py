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
            logger.info(f"Initialized LLM client with model: {self.model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'.")
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}")

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
                error_msg = str(e)
                # 检查是否为配额或频率限制错误
                if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                    logger.warning(f"LLM Rate limit/Quota exceeded: {e}")

                retries += 1
                if retries > max_retries:
                    logger.error(f"LLM call failed after {max_retries} retries: {e}")
                    raise LLMError(f"LLM call failed after {max_retries} retries: {e}")

                wait_time = initial_backoff * (2 ** (retries - 1))
                logger.warning(f"LLM call attempt {retries} failed: {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)

        raise LLMError("Unexpected exit from retry loop.")

    def ask(self, query: str, context: str) -> str:
        """基于上下文回答问题，带有更完善的系统提示词"""
        system_prompt = (
            "你是一名资深的金融风险分析专家。请根据提供的上下文内容，严谨、专业地回答用户问题。"
            "如果上下文中没有相关信息，请明确告知，不要编造。回答应包含关键风险点分析和结论建议。"
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"【参考上下文】\n{context}\n\n【用户问题】\n{query}\n\n请开始分析：",
            },
        ]
        return self.chat(messages)
