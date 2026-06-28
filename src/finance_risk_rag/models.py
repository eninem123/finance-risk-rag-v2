"""
Finance-Risk-RAG 数据模型模块
============================

定义系统中通用的 Pydantic 数据模型。
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """风险实体数据模型"""

    type: str
    text: str
    risk_score: int
    confidence: float
    context: str = ""
    source: str = "rule"
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """兼容旧版字典转换"""
        return self.model_dump()

    @property
    def key(self) -> Tuple[str, str, int]:
        """实体唯一键（用于去重）"""
        return (self.type, self.text, self.start_char)


class ExtractionResult(BaseModel):
    """提取结果数据模型"""

    entities: List[Entity]
    total_risk_score: int
    risk_level: str
    extraction_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """兼容旧版字典转换"""
        res = self.model_dump()
        res["extracted_at"] = res.pop("extraction_time")
        res["total_entities"] = len(self.entities)
        return res


class ChunkConfig(BaseModel):
    """文本分块配置"""

    chunk_size: int = 800
    overlap: int = 100


class DocumentChunk(BaseModel):
    """文档分块数据模型"""

    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """查询结果数据模型"""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """文档分类结果"""

    type: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
