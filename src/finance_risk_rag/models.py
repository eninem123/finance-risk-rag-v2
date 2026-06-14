"""
Finance-Risk-RAG 数据模型模块
============================

定义系统中通用的数据类（Data Classes）。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple


@dataclass
class Entity:
    """风险实体数据类"""

    type: str
    text: str
    risk_score: int
    confidence: float
    start_char: int = -1
    end_char: int = -1
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
            "start_char": self.start_char,
            "end_char": self.end_char,
            "context": self.context,
            "source": self.source,
            **self.metadata,
        }

    @property
    def key(self) -> Tuple[str, str, int]:
        """实体唯一键（用于去重）"""
        return (self.text, self.type, self.start_char)


@dataclass
class ExtractionResult:
    """提取结果数据类"""

    entities: List[Entity]
    total_risk_score: int
    risk_level: str
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "v2.2"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "extracted_at": self.extraction_time,
            "model_version": self.model_version,
            "total_entities": len(self.entities),
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "entities": [e.to_dict() for e in self.entities],
            **self.metadata,
        }


@dataclass
class ChunkConfig:
    """文本分块配置"""

    chunk_size: int = 800
    overlap: int = 100


@dataclass
class DocumentChunk:
    """文档分块数据类"""

    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """查询结果数据类"""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """文档分类结果"""

    type: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "confidence": self.confidence, "reason": self.reason}
