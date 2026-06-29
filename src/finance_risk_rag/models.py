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
    context: str = ""
    source: str = "rule"
    start_char: int = 0
    end_char: int = 0
    risk_category: str = "其他"  # 风险分类，如：信用风险、合规风险、经营风险
    impact_score: float = 1.0  # 影响程度评分 (1.0 - 5.0)
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
            "start_char": self.start_char,
            "end_char": self.end_char,
            "risk_category": self.risk_category,
            "impact_score": round(self.impact_score, 2),
            **self.metadata,
        }

    @property
    def key(self) -> Tuple[str, str, int]:
        """实体唯一键（用于去重）"""
        return (self.type, self.text, self.start_char)


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
