"""
Finance-Risk-RAG Configuration Module
=====================================

Centralized management of system configuration parameters with environment variable support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """
    System Configuration Class

    All configuration items can be overridden by environment variables.
    """

    # ==================== Path Configuration ====================

    # Project root directory
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())

    # Chroma vector database path
    chroma_db_dir: Path = field(default_factory=lambda: Path("rag_db"))

    # Cache and Log directories
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # Data directories
    docs_dir: Path = field(default_factory=lambda: Path("docs"))
    knowledge_base_dir: Path = field(default_factory=lambda: Path("knowledge_base"))

    # ==================== LLM Configuration ====================

    llm_provider: str = "moonshot"
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.moonshot.cn/v1"
    llm_model_name: str = "moonshot-v1-8k"
    max_context_tokens: int = 2000

    # ==================== OCR Configuration ====================

    tesseract_cmd: Optional[str] = None
    ocr_languages: str = "chi_tra+chi_sim+eng"
    ocr_dpi: int = 600
    ocr_version: str = "v2.0"

    # ==================== Risk Assessment ====================

    risk_level_low: int = 30
    risk_level_medium: int = 60
    risk_level_high: int = 90

    # ==================== Processing Configuration ====================

    chunk_size: int = 800
    chunk_overlap: int = 100
    batch_size: int = 100

    def __post_init__(self) -> None:
        """Initialize and load from environment variables."""
        self._load_from_env()
        self._resolve_paths()
        self.ensure_directories()

    def _load_from_env(self) -> None:
        """Override defaults with environment variables if present."""
        self.llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_base_url = os.getenv("LLM_BASE_URL", self.llm_base_url)
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", self.llm_model_name)

        tess_path = os.getenv("TESSERACT_CMD")
        if tess_path:
            self.tesseract_cmd = tess_path

    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute paths based on project root."""
        for attr in ["chroma_db_dir", "cache_dir", "log_dir", "docs_dir", "knowledge_base_dir"]:
            path = getattr(self, attr)
            if not path.is_absolute():
                setattr(self, attr, self.base_dir / path)

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        for attr in ["chroma_db_dir", "cache_dir", "log_dir", "docs_dir", "knowledge_base_dir"]:
            getattr(self, attr).mkdir(parents=True, exist_ok=True)

    @property
    def risk_entities_path(self) -> Path:
        return self.knowledge_base_dir / "risk_entities.json"

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
