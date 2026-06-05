"""
Configuration management for the Finance-Risk-RAG system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """
    System configuration class.

    Loads settings from environment variables and provides sensible defaults.
    """

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())
    chroma_db_dir: Path = field(default_factory=lambda: Path("rag_db"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    docs_dir: Path = field(default_factory=lambda: Path("docs"))
    knowledge_base_dir: Path = field(default_factory=lambda: Path("knowledge_base"))

    # LLM
    llm_provider: str = "moonshot"
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.moonshot.cn/v1"
    llm_model_name: str = "moonshot-v1-8k"
    max_context_tokens: int = 2000

    # Embeddings
    embedding_backend: str = "onnx"

    # OCR
    tesseract_cmd: Optional[str] = None
    ocr_languages: str = "chi_tra+chi_sim+eng"
    ocr_dpi: int = 600
    ocr_version: str = "v7"

    # Risk
    risk_level_low: int = 30
    risk_level_medium: int = 60
    risk_level_high: int = 90

    # Processing
    chunk_size: int = 800
    chunk_overlap: int = 100
    batch_size: int = 100

    def __post_init__(self) -> None:
        self._load_from_env()
        self._resolve_paths()

    def _load_from_env(self) -> None:
        """Override defaults with environment variables."""
        self.llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_base_url = os.getenv("LLM_BASE_URL", self.llm_base_url)
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", self.llm_model_name)

        self.tesseract_cmd = os.getenv("TESSERACT_CMD", self.tesseract_cmd)

        # Numeric values
        if os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(os.getenv("CHUNK_SIZE", 800))
        if os.getenv("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute paths based on base_dir."""
        for attr in ["chroma_db_dir", "cache_dir", "log_dir", "docs_dir", "knowledge_base_dir"]:
            path = getattr(self, attr)
            if not path.is_absolute():
                setattr(self, attr, self.base_dir / path)

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
        """Create necessary directories if they don't exist."""
        for attr in ["chroma_db_dir", "cache_dir", "log_dir", "docs_dir", "knowledge_base_dir"]:
            getattr(self, attr).mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Check if essential configuration is present."""
        if not self.llm_api_key:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_provider": self.llm_provider,
            "llm_model_name": self.llm_model_name,
            "chunk_size": self.chunk_size,
            "ocr_version": self.ocr_version,
        }


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
