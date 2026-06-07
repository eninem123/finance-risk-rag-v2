class FinanceRiskRAGError(Exception):
    """Finance-Risk-RAG 基础异常类"""

    pass


class ConfigError(FinanceRiskRAGError):
    """配置相关异常"""

    pass


class UtilsError(FinanceRiskRAGError):
    """工具模块基础异常"""

    pass


class FileOperationError(UtilsError):
    """文件操作异常"""

    pass


class TextProcessingError(UtilsError):
    """文本处理异常"""

    pass


class OCRError(FinanceRiskRAGError):
    """OCR 处理相关异常"""

    pass


class RAGError(FinanceRiskRAGError):
    """RAG 系统基础异常"""

    pass


class EmbeddingError(RAGError):
    """嵌入模型相关异常"""

    pass


class LLMError(RAGError):
    """LLM 调用相关异常"""

    pass


class DatabaseError(RAGError):
    """数据库相关异常"""

    pass


class ExtractionError(FinanceRiskRAGError):
    """实体提取异常"""

    pass


class RuleLoadError(ExtractionError):
    """规则加载异常"""

    pass
