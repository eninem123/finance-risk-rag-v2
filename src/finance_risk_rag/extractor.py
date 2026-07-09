"""
Finance-Risk-RAG 实体提取模块
============================
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .config import get_config
from .exceptions import ExtractionError
from .models import Entity, ExtractionResult
from .utils import calculate_risk_level, clean_text, load_json_file

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """风险评分策略基类"""

    @abstractmethod
    def calculate(self, entity: Entity) -> float:
        """计算实体的最终影响分数"""
        pass


class FinanceRiskScoringStrategy(ScoringStrategy):
    """金融风险评分策略（基于置信度和实体权重）"""

    def calculate(self, entity: Entity) -> float:
        # 基础评分 * 置信度
        impact = entity.risk_score * entity.confidence

        # 根据来源调整
        if entity.source == "bert":
            impact *= 1.2  # BERT 提取的实体通常具有更高的上下文相关性

        return round(impact, 2)


class RuleBasedExtractor:
    """基于规则的实体提取器"""

    def __init__(self, config=None, rules_path: Optional[Path] = None):
        self.config = config or get_config()
        self.rules = {}
        if rules_path:
            self.load_rules(rules_path)
        elif self.config.risk_entities_path.exists():
            self.load_rules(self.config.risk_entities_path)

    def load_rules(self, rules_path: Path):
        try:
            self.rules = load_json_file(rules_path)
            logger.info(f"Loaded {len(self.rules)} rule categories.")
        except Exception as e:
            raise ExtractionError(f"Failed to load rules: {e}")

    def extract(self, text: str) -> List[Entity]:
        if not text or not self.rules:
            return []

        entities: List[Entity] = []
        seen: Set[Tuple[str, str, int]] = set()

        for entity_type, config in self.rules.items():
            keywords = config.get("keywords", [])
            base_risk_score = config.get("risk_score", 10)

            for keyword in keywords:
                # Use regex for better matching, but avoid \b for CJK
                # Basic implementation: find all occurrences
                for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                    start = match.start()
                    end = match.end()
                    key = (entity_type, keyword, start)
                    if key in seen:
                        continue
                    seen.add(key)

                    context_start = max(0, start - 80)
                    context_end = min(len(text), end + 80)
                    context = text[context_start:context_end].replace("\n", " ").strip()

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=keyword,
                            risk_score=base_risk_score,
                            confidence=1.0,
                            context=context,
                            source="rule",
                            start_char=start,
                            end_char=end,
                        )
                    )
        return entities


class BERTExtractor:
    """基于 BERT 的实体提取器"""

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, model_path: Path):
        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
            self.device = 0 if torch.cuda.is_available() else -1
            self.nlp = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple",
            )
            logger.info(f"BERT model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def extract(self, text: str) -> List[Entity]:
        """
        使用 BERT 提取实体，支持长文本切片。
        """
        if not self.is_available() or not text:
            return []

        # BERT 通常限制 512 tokens，我们采用滑动窗口处理
        max_length = 500
        overlap = 50
        entities = []

        try:
            chunks = self._chunk_text(text, max_length, overlap)
            for chunk_text, offset in chunks:
                results = self.nlp(chunk_text)
                for res in results:
                    start = res["start"] + offset
                    end = res["end"] + offset
                    entities.append(
                        Entity(
                            type=res["entity_group"],
                            text=res["word"],
                            risk_score=20,
                            confidence=float(res["score"]),
                            context=text[max(0, start - 40) : min(len(text), end + 40)],
                            source="bert",
                            start_char=start,
                            end_char=end,
                        )
                    )
            return self._deduplicate_entities(entities)
        except Exception as e:
            logger.error(f"BERT extraction failed: {e}")
            return []

    def _chunk_text(self, text: str, max_length: int, overlap: int) -> List[Tuple[str, int]]:
        chunks = []
        if len(text) <= max_length:
            return [(text, 0)]

        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append((text[start:end], start))
            if end == len(text):
                break
            start += max_length - overlap
        return chunks

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        if not entities:
            return []
        # 按位置和文本去重
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.type, e.text, e.start_char)
            if key not in seen:
                unique_entities.append(e)
                seen.add(key)
        return unique_entities


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(
        self,
        config=None,
        rule_extractor=None,
        bert_extractor=None,
        scoring_strategy: Optional[ScoringStrategy] = None,
    ):
        self.config = config or get_config()
        self.rule_extractor = rule_extractor or RuleBasedExtractor(config=self.config)
        self.bert_extractor = bert_extractor or BERTExtractor(self.config.bert_local_path)
        self.scoring_strategy = scoring_strategy or FinanceRiskScoringStrategy()

    def process(self, text_or_path: Union[str, Path]) -> ExtractionResult:
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            text = text_or_path

        text = clean_text(text)

        rule_entities = self.rule_extractor.extract(text)
        bert_entities = self.bert_extractor.extract(text)

        # Advanced merging logic: Score-based arbitration and overlap resolution
        entities_list = self._merge_and_arbitrate(rule_entities, bert_entities)

        # 计算最终影响分数
        for entity in entities_list:
            entity.impact_score = self.scoring_strategy.calculate(entity)

        total_risk = sum(e.risk_score for e in entities_list)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=entities_list, total_risk_score=total_risk, risk_level=risk_level
        )

    def _merge_and_arbitrate(
        self, rule_entities: List[Entity], bert_entities: List[Entity]
    ) -> List[Entity]:
        """
        合并规则引擎和 BERT 的结果，处理重叠。
        仲裁机制：
        1. 如果 BERT 实体的置信度 > 0.85，在发生冲突时优先保留 BERT 结果。
        2. 否则，保留得分和置信度更高的实体。
        """
        if not rule_entities and not bert_entities:
            return []

        # 分离出高置信度的 BERT 实体
        high_conf_bert = [e for e in bert_entities if e.confidence > 0.85]
        other_entities = rule_entities + [e for e in bert_entities if e.confidence <= 0.85]

        # 先处理高置信度 BERT
        high_conf_bert.sort(key=lambda x: x.confidence, reverse=True)
        final_entities: List[Entity] = []

        for current in high_conf_bert:
            overlap = False
            for existing in final_entities:
                if max(current.start_char, existing.start_char) < min(
                    current.end_char, existing.end_char
                ):
                    overlap = True
                    break
            if not overlap:
                final_entities.append(current)

        # 再合并其他实体
        other_entities.sort(key=lambda x: (x.risk_score, x.confidence), reverse=True)
        for current in other_entities:
            is_redundant = False
            for existing in final_entities:
                if max(current.start_char, existing.start_char) < min(
                    current.end_char, existing.end_char
                ):
                    is_redundant = True
                    break
            if not is_redundant:
                final_entities.append(current)

        return final_entities
