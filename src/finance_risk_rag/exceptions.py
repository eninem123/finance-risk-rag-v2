"""
Custom exceptions for the Finance-Risk-RAG system.
"""


class FinanceRiskRAGError(Exception):
    """Base exception for all system errors."""

    pass


class ConfigError(FinanceRiskRAGError):
    """Configuration related errors."""

    pass


class OCRError(FinanceRiskRAGError):
    """OCR processing related errors."""

    pass


class ExtractionError(FinanceRiskRAGError):
    """Entity extraction related errors."""

    pass


class RuleLoadError(ExtractionError):
    """Errors loading extraction rules."""

    pass


class RAGError(FinanceRiskRAGError):
    """Base error for RAG engine operations."""

    pass


class EmbeddingError(RAGError):
    """Errors related to embedding generation."""

    pass


class DatabaseError(RAGError):
    """Errors related to vector database operations."""

    pass


class LLMError(FinanceRiskRAGError):
    """Errors related to LLM API calls."""

    pass
