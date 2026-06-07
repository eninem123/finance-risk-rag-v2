import logging
import time
from typing import Any, Optional

from finance_risk_rag.config import get_config
from finance_risk_rag.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """LLM客户端封装类"""

    DEFAULT_MODEL = "moonshot-v1-8k"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 512
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        初始化LLM客户端
        """
        config = get_config()
        self._api_key = api_key or config.llm_api_key
        self._base_url = base_url or config.llm_base_url
        self._model_name = model_name or config.llm_model_name or self.DEFAULT_MODEL
        self._client: Optional[Any] = None

        if not self._api_key:
            logger.warning("未检测到 LLM API key。")
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            logger.info(f"LLM客户端初始化成功，模型: {self._model_name}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise LLMError(f"无法初始化LLM客户端: {e}") from e

    @property
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self._client is not None

    def ask(
        self,
        query: str,
        context: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str = "你是一名金融风险分析顾问，回答时引用上下文并给出简明结论。",
    ) -> str:
        """
        向LLM提问，带有重试逻辑
        """
        if not self.is_available:
            raise LLMError("LLM客户端未初始化，请设置API密钥")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"参考以下上下文来回答问题：\n\n{context}\n\n问题：{query}",
            },
        ]

        last_error = None
        if self._client is None:
            raise LLMError("LLM 客户端未初始化")

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"LLM调用在 {self.MAX_RETRIES} 次尝试后失败: {last_error}")
        raise LLMError(f"LLM调用失败: {last_error}") from last_error
