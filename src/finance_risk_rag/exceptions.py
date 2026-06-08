"""
Finance-Risk-RAG 异常定义
"""

class FinanceRiskRAGError(Exception):
    """系统基础异常"""
    pass


# ==================== RAG 异常 ====================

class RAGError(FinanceRiskRAGError):
    """RAG系统基础异常"""
    pass


class EmbeddingError(RAGError):
    """嵌入模型相关异常"""
    pass


class LLMError(RAGError):
    """LLM调用相关异常"""
    pass


class DatabaseError(RAGError):
    """数据库相关异常"""
    pass


# ==================== 提取异常 ====================

class ExtractionError(FinanceRiskRAGError):
    """实体提取异常"""
    pass


class RuleLoadError(ExtractionError):
    """规则加载异常"""
    pass


# ==================== OCR 异常 ====================

class OCRError(FinanceRiskRAGError):
    """OCR处理异常"""
    pass


# ==================== 工具异常 ====================

class UtilsError(FinanceRiskRAGError):
    """工具模块基础异常"""
    pass


class FileOperationError(UtilsError):
    """文件操作异常"""
    pass


class TextProcessingError(UtilsError):
    """文本处理异常"""
    pass
