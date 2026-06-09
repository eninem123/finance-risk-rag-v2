"""
Finance-Risk-RAG 异常处理模块
============================

定义系统中使用的自定义异常类。
"""


class FinanceRiskRAGError(Exception):
    """Finance-Risk-RAG 系统的基础异常类"""

    pass


# ==================== RAG 相关异常 ====================


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


# ==================== 提取相关异常 ====================


class ExtractionError(FinanceRiskRAGError):
    """实体提取异常"""

    pass


class RuleLoadError(ExtractionError):
    """规则加载异常"""

    pass


# ==================== OCR 相关异常 ====================


class OCRError(FinanceRiskRAGError):
    """OCR 识别异常"""

    pass


# ==================== 工具相关异常 ====================


class UtilsError(FinanceRiskRAGError):
    """工具模块基础异常"""

    pass


class FileOperationError(UtilsError):
    """文件操作异常"""

    pass


class TextProcessingError(UtilsError):
    """文本处理异常"""

    pass
