"""
Finance-Risk-RAG 数据模型
========================

定义系统中使用的共享数据结构和类型。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple


class EmbeddingBackend(Enum):
    """嵌入模型后端枚举"""

    ONNX = "onnx"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class ChunkConfig:
    """文本分块配置"""

    chunk_size: int = 800
    overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.overlap < 0:
            raise ValueError("overlap 不能为负数")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")


@dataclass
class QueryResult:
    """查询结果数据类"""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """文档分块数据类"""

    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """风险实体数据类"""

    type: str
    text: str
    risk_score: int
    confidence: float
    context: str = ""
    source: str = "rule"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "text": self.text,
            "risk_score": self.risk_score,
            "confidence": round(self.confidence, 4),
            "context": self.context,
            "source": self.source,
            **self.metadata,
        }

    @property
    def key(self) -> Tuple[str, str]:
        """实体唯一键（用于去重）"""
        return (self.text, self.type)


@dataclass
class ExtractionResult:
    """提取结果数据类"""

    entities: List[Entity]
    total_risk_score: int
    risk_level: str
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "extracted_at": self.extraction_time,
            "total_entities": len(self.entities),
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "entities": [e.to_dict() for e in self.entities],
            **self.metadata,
        }
