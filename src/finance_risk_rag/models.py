"""
Shared data models for the Finance-Risk-RAG system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Entity:
    """Risk entity data class."""
    type: str
    text: str
    risk_score: int
    confidence: float
    context: str = ""
    source: str = "rule"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "text": self.text,
            "risk_score": self.risk_score,
            "confidence": round(self.confidence, 4),
            "context": self.context,
            "source": self.source,
            **self.metadata
        }

    @property
    def key(self) -> Tuple[str, str]:
        """Unique key for deduplication (text, type)."""
        return (self.text, self.type)


@dataclass
class ExtractionResult:
    """Data class for extraction results."""
    entities: List[Entity]
    total_risk_score: int
    risk_level: str
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "extracted_at": self.extraction_time,
            "total_entities": len(self.entities),
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "entities": [e.to_dict() for e in self.entities],
            **self.metadata
        }


@dataclass
class QueryResult:
    """Data class for RAG query results."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 800
    overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")


@dataclass
class DocumentChunk:
    """Data class for a document chunk."""
    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
