"""
Finance-Risk-RAG 配置模块
========================

集中管理系统配置参数，支持环境变量覆盖。

使用方法:
    from config import Config

    config = Config()
    print(config.llm_api_key)
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

    # API密钥（优先级：OPENAI_API_KEY > MOONSHOT_API_KEY）
    llm_api_key: Optional[str] = None

    # API基础URL
    llm_base_url: str = "https://api.moonshot.cn/v1"

    # 默认模型名称
    llm_model_name: str = "moonshot-v1-8k"

    # 最大上下文token数
    max_context_tokens: int = 2000

    # ==================== 嵌入模型配置 ====================

    # 嵌入模型后端: "onnx" 或 "sentence_transformers"
    embedding_backend: str = "onnx_or_sbert"

    # ==================== OCR 配置 ====================

    # Tesseract可执行文件路径
    tesseract_cmd: Optional[str] = None

    # OCR默认语言
    ocr_languages: str = "chi_tra+chi_sim+eng"

    # OCR DPI
    ocr_dpi: int = 600

    # OCR版本号（修改参数后递增）
    ocr_version: str = "v7"

    # ==================== 风险评估配置 ====================

    # 风险等级阈值
    risk_level_low: int = 30
    risk_level_medium: int = 60
    risk_level_high: int = 90

    # ==================== 处理配置 ====================

    # 文本分块大小
    chunk_size: int = 800

    # 分块重叠大小
    chunk_overlap: int = 100

    # 批量处理大小
    batch_size: int = 100

    # API调用间隔（秒）
    api_call_interval: float = 1.0

    def __post_init__(self) -> None:
        """初始化后处理，从环境变量加载配置"""
        self._load_from_env()
        self._resolve_paths()

    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        # LLM配置
        self.llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_base_url = os.getenv("LLM_BASE_URL", self.llm_base_url)
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", self.llm_model_name)
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", self.max_context_tokens))

        # 嵌入模型配置
        self.embedding_backend = os.getenv("EMBEDDING_BACKEND", self.embedding_backend)

        # OCR配置
        self.tesseract_cmd = os.getenv("TESSERACT_CMD", self.tesseract_cmd)

        # 处理配置
        self.chunk_size = int(os.getenv("CHUNK_SIZE", self.chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.chunk_overlap))

    def _resolve_paths(self) -> None:
        """解析相对路径为绝对路径"""
        # 将相对路径转换为绝对路径
        if not self.chroma_db_dir.is_absolute():
            self.chroma_db_dir = self.base_dir / self.chroma_db_dir

        if not self.cache_dir.is_absolute():
            self.cache_dir = self.base_dir / self.cache_dir

        if not self.log_dir.is_absolute():
            self.log_dir = self.base_dir / self.log_dir

        if not self.docs_dir.is_absolute():
            self.docs_dir = self.base_dir / self.docs_dir

        if not self.knowledge_base_dir.is_absolute():
            self.knowledge_base_dir = self.base_dir / self.knowledge_base_dir

        # BERT模型路径
        bert_path = self.base_dir / "hfl" / "chinese-bert-wwm-ext"
        if bert_path.exists():
            self.bert_local_path = bert_path

    @property
    def risk_entities_path(self) -> Path:
        """风险实体规则文件路径"""
        return self.knowledge_base_dir / "risk_entities.json"

    @property
    def stopwords_path(self) -> Path:
        """停用词文件路径"""
        return self.knowledge_base_dir / "stopwords.txt"

    @property
    def finance_dict_path(self) -> Path:
        """金融词典文件路径"""
        return self.knowledge_base_dir / "finance_dict.txt"

    @property
    def processing_log_path(self) -> Path:
        """处理日志文件路径"""
        return self.cache_dir / "processing_log.json"

    def ensure_directories(self) -> None:
        """确保所有必要目录存在"""
        for dir_path in [
            self.chroma_db_dir,
            self.cache_dir,
            self.log_dir,
            self.docs_dir,
            self.knowledge_base_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """
        验证配置是否有效

        Returns:
            配置是否有效
        """
        errors = []

        if not self.llm_api_key:
            errors.append("未设置LLM API密钥 (OPENAI_API_KEY 或 MOONSHOT_API_KEY)")

        if self.tesseract_cmd and not Path(self.tesseract_cmd).exists():
            errors.append(f"Tesseract路径不存在: {self.tesseract_cmd}")

        if self.chunk_overlap >= self.chunk_size:
            errors.append(f"分块重叠({self.chunk_overlap})必须小于分块大小({self.chunk_size})")

        if errors:
            for error in errors:
                print(f"配置错误: {error}")
            return False

        return True

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "base_dir": str(self.base_dir),
            "chroma_db_dir": str(self.chroma_db_dir),
            "llm_provider": self.llm_provider,
            "llm_base_url": self.llm_base_url,
            "llm_model_name": self.llm_model_name,
            "embedding_backend": self.embedding_backend,
            "ocr_dpi": self.ocr_dpi,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config()
    return _config


# 向后兼容的模块级变量
_config_instance = get_config()

# 路径配置
BASE_DIR = _config_instance.base_dir
BERT_LOCAL_PATH = _config_instance.bert_local_path
CHROMA_DB_DIR = _config_instance.chroma_db_dir

# LLM配置
LLM_PROVIDER = _config_instance.llm_provider
LLM_API_KEY = _config_instance.llm_api_key
LLM_BASE_URL = _config_instance.llm_base_url

# 嵌入模型配置
EMBEDDING_BACKEND = _config_instance.embedding_backend

# OCR配置
TESSERACT_CMD = _config_instance.tesseract_cmd

# 处理配置
MAX_CONTEXT_TOKENS = _config_instance.max_context_tokens


if __name__ == "__main__":
    config = get_config()
    print("配置信息:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    print(f"\n配置验证: {'通过' if config.validate() else '失败'}")
