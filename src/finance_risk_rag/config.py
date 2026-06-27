"""
Finance-Risk-RAG 配置模块
========================

集中管理系统配置参数，支持环境变量覆盖。
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    系统配置类，使用 Pydantic Settings 进行管理。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ==================== 路径配置 ====================

    # 项目根目录
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())

    # BERT本地模型路径
    bert_local_path: Optional[Path] = None

    # Chroma向量数据库路径
    chroma_db_dir: Path = Field(default=Path("rag_db"))

    # 缓存目录
    cache_dir: Path = Field(default=Path("cache"))

    # 日志目录
    log_dir: Path = Field(default=Path("logs"))

    # 文档目录
    docs_dir: Path = Field(default=Path("docs"))

    # 知识库目录
    knowledge_base_dir: Path = Field(default=Path("knowledge_base"))

    # ==================== LLM 配置 ====================

    llm_provider: str = "moonshot"
    llm_api_key: Optional[str] = Field(default=None, validation_alias="MOONSHOT_API_KEY")
    llm_base_url: str = "https://api.moonshot.cn/v1"
    llm_model_name: str = "moonshot-v1-8k"
    max_context_tokens: int = 2000

    # ==================== 嵌入模型配置 ====================

    embedding_backend: str = "onnx"

    # ==================== OCR 配置 ====================

    tesseract_cmd: Optional[str] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ocr_languages: str = "chi_tra+chi_sim+eng"
    ocr_dpi: int = 600
    ocr_version: str = "v7"

    # ==================== 风险评估配置 ====================

    risk_level_low: int = 30
    risk_level_medium: int = 60
    risk_level_high: int = 90

    # ==================== 处理配置 ====================

    chunk_size: int = 800
    chunk_overlap: int = 100
    batch_size: int = 100

    def model_post_init(self, __context) -> None:
        """从环境变量加载配置并解析路径"""
        # Pydantic 自动处理环境变量，这里主要处理路径解析和 BERT 路径
        self._resolve_paths()

        # 兼容旧的 API Key 环境变量名
        if not self.llm_api_key:
            self.llm_api_key = os.getenv("OPENAI_API_KEY")

    def _resolve_paths(self) -> None:
        # 将相对路径转换为绝对路径
        for attr in ["chroma_db_dir", "cache_dir", "log_dir", "docs_dir", "knowledge_base_dir"]:
            path_val = getattr(self, attr)
            if not path_val.is_absolute():
                setattr(self, attr, self.base_dir / path_val)

        # BERT模型路径
        bert_path = self.base_dir / "hfl" / "chinese-bert-wwm-ext"
        if bert_path.exists():
            self.bert_local_path = bert_path

    @property
    def risk_entities_path(self) -> Path:
        return self.knowledge_base_dir / "risk_entities.json"

    @property
    def stopwords_path(self) -> Path:
        return self.knowledge_base_dir / "stopwords.txt"

    @property
    def finance_dict_path(self) -> Path:
        return self.knowledge_base_dir / "finance_dict.txt"

    @property
    def processing_log_path(self) -> Path:
        return self.cache_dir / "processing_log.json"

    def ensure_directories(self) -> None:
        for dir_path in [
            self.chroma_db_dir,
            self.cache_dir,
            self.log_dir,
            self.docs_dir,
            self.knowledge_base_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
