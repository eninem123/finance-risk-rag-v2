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
    """提取实体风险评分策略基类"""

    @abstractmethod
    def calculate_score(self, entity_type: str, confidence: float, **kwargs) -> int:
        """计算实体的风险分值"""
        pass


class FinanceRiskScoringStrategy(ScoringStrategy):
    """金融风险量化评分策略"""

    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or {}

    def calculate_score(self, entity_type: str, confidence: float, **kwargs) -> int:
        # 基础分数来自于规则配置，默认为 10
        rule_config = self.rules.get(entity_type, {})
        base_score = rule_config.get("risk_score", 20 if "bert" in kwargs.get("source", "") else 10)
        # 根据置信度进行加权调整
        return int(base_score * confidence)


class BaseExtractor(ABC):
    """实体提取器基类"""

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """从文本中提取实体"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """检查提取器是否可用"""
        pass


class RuleBasedExtractor(BaseExtractor):
    """基于规则的实体提取器"""

    @property
    def is_available(self) -> bool:
        return bool(self.rules)

    def __init__(
        self,
        config=None,
        rules_path: Optional[Path] = None,
        scoring_strategy: Optional[ScoringStrategy] = None,
    ):
        self.config = config or get_config()
        self.rules = {}
        if rules_path:
            self.load_rules(rules_path)
        elif self.config.risk_entities_path.exists():
            self.load_rules(self.config.risk_entities_path)

        self.scoring_strategy = scoring_strategy or FinanceRiskScoringStrategy(self.rules)

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

            for keyword in keywords:
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

                    risk_score = self.scoring_strategy.calculate_score(
                        entity_type, 1.0, source="rule"
                    )

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=keyword,
                            risk_score=risk_score,
                            confidence=1.0,
                            context=context,
                            source="rule",
                            start_char=start,
                            end_char=end,
                        )
                    )
        return entities


class BERTExtractor(BaseExtractor):
    """基于 BERT 的实体提取器"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scoring_strategy: Optional[ScoringStrategy] = None,
    ):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.scoring_strategy = scoring_strategy or FinanceRiskScoringStrategy()
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, model_path: Path):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

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

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available or not text:
            return []

        try:
            # 针对长文本使用滑动窗口切片
            segments = self._chunk_text(text)
            all_entities = []

            for segment_text, offset in segments:
                results = self.nlp(segment_text)
                for res in results:
                    confidence = float(res["score"])
                    risk_score = self.scoring_strategy.calculate_score(
                        res["entity_group"], confidence, source="bert"
                    )

                    all_entities.append(
                        Entity(
                            type=res["entity_group"],
                            text=res["word"],
                            risk_score=risk_score,
                            confidence=confidence,
                            context=segment_text[
                                max(0, res["start"] - 40) : min(len(segment_text), res["end"] + 40)
                            ],
                            source="bert",
                            start_char=res["start"] + offset,
                            end_char=res["end"] + offset,
                        )
                    )
            return all_entities
        except Exception as e:
            logger.error(f"BERT extraction failed: {e}")
            return []

    def _chunk_text(
        self, text: str, max_length: int = 512, overlap: int = 50
    ) -> List[Tuple[str, int]]:
        """将长文本切分为带重叠的片段以适应 BERT 限制"""
        if len(text) <= max_length:
            return [(text, 0)]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append((text[start:end], start))
            if end == len(text):
                break
            start += max_length - overlap
        return chunks


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(self, config=None, rule_extractor=None, bert_extractor=None):
        self.config = config or get_config()
        self.rule_extractor = rule_extractor or RuleBasedExtractor(config=self.config)
        self.bert_extractor = bert_extractor or BERTExtractor(self.config.bert_local_path)

    def process(self, text_or_path: Union[str, Path]) -> ExtractionResult:
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            text = text_or_path

        text = clean_text(text)

        rule_entities = self.rule_extractor.extract(text)
        bert_entities = self.bert_extractor.extract(text)

        # 高级合并逻辑：基于评分的仲裁与重叠消除
        entities_list = self._merge_and_arbitrate(rule_entities, bert_entities)

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
        优先规则：
        1. 优先保留 BERT 置信度 > 0.85 的实体。
        2. 在重叠情况下，保留风险评分较高的实体。
        """
        all_entities = rule_entities + bert_entities
        if not all_entities:
            return []

        # 排序优先级：BERT 高置信度 (>0.85) > 风险评分 > 置信度
        def sort_key(e: Entity):
            is_high_conf_bert = 1 if (e.source == "bert" and e.confidence > 0.85) else 0
            return (is_high_conf_bert, e.risk_score, e.confidence)

        all_entities.sort(key=sort_key, reverse=True)

        final_entities: List[Entity] = []

        for current in all_entities:
            is_redundant = False
            for existing in final_entities:
                # 精确的重叠检测
                overlap = max(current.start_char, existing.start_char) < min(
                    current.end_char, existing.end_char
                )
                if overlap:
                    is_redundant = True
                    break

            if not is_redundant:
                final_entities.append(current)

        # 最终按起始位置排序，方便展示
        final_entities.sort(key=lambda x: x.start_char)
        return final_entities
