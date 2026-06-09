"""
Finance-Risk-RAG 实体提取模块
============================

从财务文档中提取风险实体，支持规则提取和BERT模型提取。
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import get_config
from .exceptions import ExtractionError, RuleLoadError
from .models import Entity, ExtractionResult
from .utils import (
    calculate_risk_level,
    clean_text,
    load_json_file,
    save_json_file,
    setup_logger,
)

# ==================== 规则实体提取器 ====================


class RuleBasedExtractor:
    """基于规则的实体提取器"""

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        """
        初始化规则提取器

        Args:
            rules_path: 规则文件路径
        """
        self._rules: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: Path) -> None:
        """
        加载实体规则

        Args:
            rules_path: 规则文件路径
        """
        try:
            self._rules = load_json_file(rules_path)
            self._logger.info(f"加载规则成功: {len(self._rules)} 类")
        except Exception as e:
            self._logger.error(f"规则加载失败: {e}")
            raise RuleLoadError(f"无法加载规则文件: {e}") from e

    def extract(self, text: str) -> List[Entity]:
        """
        从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            提取的实体列表
        """
        if not text or not self._rules:
            return []

        entities: List[Entity] = []
        seen: Set[Tuple[str, str, int]] = set()

        # 关键词匹配
        for entity_type, config in self._rules.items():
            keywords = config.get("keywords", [])
            base_risk_score = config.get("risk_score", 10)

            for keyword in keywords:
                # 优化正则：支持中英文边界
                pattern = rf"(?:^|(?<=[^\w]))({re.escape(keyword)})(?:(?=[^\w])|$)"

                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start(1)
                    key = (entity_type, keyword, start)

                    if key in seen:
                        continue
                    seen.add(key)

                    # 提取上下文
                    context_start = max(0, start - 80)
                    context_end = min(len(text), start + len(keyword) + 80)
                    context = text[context_start:context_end].replace("\n", " ").strip()

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=keyword,
                            risk_score=base_risk_score,
                            confidence=1.0,
                            context=context,
                            source="rule",
                        )
                    )

        return entities


# ==================== BERT 实体提取器 ====================


class BERTExtractor:
    """基于BERT的实体提取器"""

    # 风险评分映射
    RISK_SCORE_MAP = {"RISK": 30, "MONEY": 25, "ORG": 15, "PER": 5, "LOC": 5}

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        初始化BERT提取器

        Args:
            model_path: 模型路径
        """
        self._pipeline: Any = None
        self._logger = logging.getLogger(__name__)

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Path) -> bool:
        """
        加载BERT模型

        Args:
            model_path: 模型路径
        """
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "token-classification",
                model=str(model_path),
                tokenizer=str(model_path),
                aggregation_strategy="simple",
            )

            self._logger.info(f"BERT模型加载成功: {model_path}")
            return True
        except Exception as e:
            self._logger.warning(f"BERT模型加载失败: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self._pipeline is not None

    def extract(self, text: str, max_length: int = 512, overlap: int = 50) -> List[Entity]:
        """
        使用BERT提取实体

        Args:
            text: 输入文本
            max_length: 最大长度
            overlap: 重叠大小

        Returns:
            提取的实体列表
        """
        if not self.is_available or not text:
            return []

        entities: List[Entity] = []
        chunks = self._chunk_text(text, max_length, overlap)

        for i, chunk in enumerate(chunks):
            try:
                chunk_entities = self._extract_from_chunk(chunk)
                entities.extend(chunk_entities)
            except Exception as e:
                self._logger.warning(f"分块 {i} 提取失败: {e}")

        return entities

    def _chunk_text(self, text: str, max_length: int, overlap: int) -> List[str]:
        """简单的按字符分块"""
        chunks = []
        if len(text) <= max_length:
            return [text]

        start = 0
        while start < len(text):
            end = start + max_length
            chunks.append(text[start:end])
            start += max_length - overlap
            if end >= len(text):
                break
        return chunks

    def _extract_from_chunk(self, chunk: str) -> List[Entity]:
        """从单个分块提取实体"""
        if self._pipeline is None:
            return []

        results: List[Dict[str, Any]] = self._pipeline(chunk)
        entities = []

        for res in results:
            entity_group = res.get("entity_group", "UNKNOWN")
            word = res.get("word", "")
            score = float(res.get("score", 0.0))
            start = res.get("start", 0)

            if score < 0.7:  # 置信度阈值
                continue

            risk_score = self.RISK_SCORE_MAP.get(entity_group, 10)

            # 提取上下文
            context_start = max(0, start - 40)
            context_end = min(len(chunk), start + len(word) + 40)
            context = chunk[context_start:context_end].strip()

            entities.append(
                Entity(
                    type=entity_group,
                    text=word,
                    risk_score=risk_score,
                    confidence=score,
                    context=context,
                    source="bert",
                )
            )

        return entities


# ==================== 实体融合器 ====================


class EntityMerger:
    """实体融合器"""

    def merge(self, rule_entities: List[Entity], bert_entities: List[Entity]) -> List[Entity]:
        """
        融合规则提取和BERT提取的实体
        """
        merged: Dict[Tuple[str, str], Entity] = {}

        for entity in rule_entities + bert_entities:
            key = entity.key

            if key not in merged:
                merged[key] = entity
            else:
                existing = merged[key]
                # 规则通常更准确（关键词匹配），BERT 辅助发现
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.risk_score = max(existing.risk_score, entity.risk_score)
                if entity.source == "rule":
                    existing.source = "rule"  # 优先标记为 rule

        return list(merged.values())


# ==================== 实体提取管道 ====================


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(self, config=None) -> None:
        self._config = config or get_config()
        self._logger = setup_logger(
            "entity_extraction", self.config.log_dir / "extract_entities.log"
        )

        self._rule_extractor = RuleBasedExtractor()
        self._bert_extractor = BERTExtractor()
        self._merger = EntityMerger()

    @property
    def config(self):
        return self._config

    def initialize(self) -> None:
        """初始化管道组件"""
        # 加载规则
        rules_path = self.config.risk_entities_path
        if rules_path.exists():
            self._rule_extractor.load_rules(rules_path)

        # 加载BERT模型
        if self.config.bert_local_path:
            self._bert_extractor.load_model(self.config.bert_local_path)

        self._logger.info("实体提取管道初始化完成")

    def process(self, text_path: Path) -> ExtractionResult:
        """
        处理文本文件
        """
        self._logger.info(f"开始处理: {text_path}")

        if not text_path.exists():
            raise ExtractionError(f"文本文件不存在: {text_path}")

        text = text_path.read_text(encoding="utf-8")
        text = clean_text(text)

        # 规则提取
        rule_entities = self._rule_extractor.extract(text)

        # BERT提取
        bert_entities = []
        if self._bert_extractor.is_available:
            bert_entities = self._bert_extractor.extract(text)

        # 融合
        final_entities = self._merger.merge(rule_entities, bert_entities)

        # 计算风险
        total_risk = sum(e.risk_score for e in final_entities)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=final_entities,
            total_risk_score=total_risk,
            risk_level=risk_level,
            metadata={
                "rule_entities_count": len(rule_entities),
                "bert_entities_count": len(bert_entities),
            },
        )

    def save_result(self, result: ExtractionResult, output_path: Path) -> None:
        """保存提取结果"""
        save_json_file(result.to_dict(), output_path)
        self._logger.info(f"结果已保存: {output_path}")
