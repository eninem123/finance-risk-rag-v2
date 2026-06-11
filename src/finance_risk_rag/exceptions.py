"""
Finance-Risk-RAG 异常定义模块
============================

集中管理系统自定义异常，确保错误处理的一致性。
"""


class FinanceRiskRAGError(Exception):
    """系统基础异常"""

    pass


class ConfigError(FinanceRiskRAGError):
    """配置相关异常"""

    pass


class OCRError(FinanceRiskRAGError):
    """OCR 处理相关异常"""

    pass


class ExtractionError(FinanceRiskRAGError):
    """实体提取相关异常"""

    pass


class RAGError(FinanceRiskRAGError):
    """RAG 引擎相关异常"""

    pass


class LLMError(FinanceRiskRAGError):
    """LLM 调用相关异常"""

    pass


class DatabaseError(FinanceRiskRAGError):
    """向量数据库相关异常"""

    pass


class FileOperationError(FinanceRiskRAGError):
    """文件操作相关异常"""

    pass
