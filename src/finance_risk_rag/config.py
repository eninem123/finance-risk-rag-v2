"""
Finance-Risk-RAG 配置模块
========================

集中管理系统配置参数，支持环境变量覆盖。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """
    系统配置类

    所有配置项支持通过环境变量覆盖，环境变量名称与属性名相同（大写）。
    """

    # ==================== 路径配置 ====================

    # 项目根目录
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())

    # BERT本地模型路径
    bert_local_path: Optional[Path] = None

    # Chroma向量数据库路径
    chroma_db_dir: Path = field(default_factory=lambda: Path("rag_db"))

    # 缓存目录
    cache_dir: Path = field(default_factory=lambda: Path("cache"))

    # 日志目录
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # 文档目录
    docs_dir: Path = field(default_factory=lambda: Path("docs"))

    # 知识库目录
    knowledge_base_dir: Path = field(default_factory=lambda: Path("knowledge_base"))

    # ==================== LLM 配置 ====================

    # LLM提供商: "moonshot" 或 "openai"
    llm_provider: str = "moonshot"

    # API密钥
    llm_api_key: Optional[str] = None

    # API基础URL
    llm_base_url: str = "https://api.moonshot.cn/v1"

    # 默认模型名称
    llm_model_name: str = "moonshot-v1-8k"

    # 最大上下文token数
    max_context_tokens: int = 2000

    # ==================== 嵌入模型配置 ====================

    # 嵌入模型后端: "onnx" 或 "sentence_transformers"
    embedding_backend: str = "onnx"

    # ==================== OCR 配置 ====================

    # Tesseract可执行文件路径
    tesseract_cmd: Optional[str] = None

    # OCR默认语言
    ocr_languages: str = "chi_tra+chi_sim+eng"

    # OCR DPI
    ocr_dpi: int = 600

    # OCR版本号
    ocr_version: str = "v7"

    # ==================== 处理配置 ====================

    # 文本分块大小
    chunk_size: int = 800

    # 分块重叠大小
    chunk_overlap: int = 100

    def __post_init__(self) -> None:
        """初始化后处理，从环境变量加载配置"""
        self._load_from_env()
        self._resolve_paths()

    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        self.llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_base_url = os.getenv("LLM_BASE_URL", self.llm_base_url)
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", self.llm_model_name)

        if os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        if os.getenv("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    def _resolve_paths(self) -> None:
        """解析相对路径为绝对路径"""
        for attr in ["chroma_db_dir", "cache_dir", "log_dir", "docs_dir", "knowledge_base_dir"]:
            val = getattr(self, attr)
            if not val.is_absolute():
                setattr(self, attr, self.base_dir / val)

    @property
    def risk_entities_path(self) -> Path:
        return self.knowledge_base_dir / "risk_entities.json"

    def validate(self) -> bool:
        return self.llm_api_key is not None


_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config()
    return _config
