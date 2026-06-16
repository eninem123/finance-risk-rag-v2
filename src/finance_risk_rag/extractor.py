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
    """评分策略基类"""

    @abstractmethod
    def calculate_score(self, entity_type: str, confidence: float, base_score: int) -> int:
        pass


class DefaultScoringStrategy(ScoringStrategy):
    """默认评分策略"""

    def calculate_score(self, entity_type: str, confidence: float, base_score: int) -> int:
        return int(base_score * confidence)


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
                            start_char=start,
                            end_char=end,
                            context=context,
                            source="rule",
                        )
                    )
        return entities


class BERTExtractor:
    """基于 BERT 的实体提取器"""

    def __init__(self, model_path: Optional[Path] = None, config=None):
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.nlp = None
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
        return self.nlp is not None

    def _chunk_text(
        self, text: str, max_length: int = 512, overlap: int = 50
    ) -> List[Tuple[str, int]]:
        """将长文本切分成块，并记录偏移量"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            chunks.append((text[start:end], start))
            start += max_length - overlap
        return chunks

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available or not text:
            return []

        try:
            # 处理长文本
            chunks = self._chunk_text(text)
            entities = []
            seen_keys = set()

            for chunk_text, offset in chunks:
                results = self.nlp(chunk_text)
                for res in results:
                    start_char = res["start"] + offset
                    end_char = res["end"] + offset
                    word = res["word"]
                    entity_type = res["entity_group"]

                    key = (entity_type, word, start_char)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=word,
                            risk_score=20,
                            confidence=float(res["score"]),
                            start_char=start_char,
                            end_char=end_char,
                            context=text[max(0, start_char - 40) : min(len(text), end_char + 40)],
                            source="bert",
                        )
                    )
            return entities
        except Exception as e:
            logger.error(f"BERT extraction failed: {e}")
            return []


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
        self.bert_extractor = bert_extractor or BERTExtractor(
            self.config.bert_local_path, config=self.config
        )
        self.scoring_strategy = scoring_strategy or DefaultScoringStrategy()

    def process(self, text_or_path: Union[str, Path]) -> ExtractionResult:
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            text = text_or_path

        text = clean_text(text)

        rule_entities = self.rule_extractor.extract(text)
        bert_entities = self.bert_extractor.extract(text)

        # 应用评分策略
        for entity in bert_entities:
            entity.risk_score = self.scoring_strategy.calculate_score(
                entity.type, entity.confidence, entity.risk_score
            )

        # 合并与仲裁
        entities_list = self._merge_and_arbitrate(rule_entities, bert_entities)

        total_risk = sum(e.risk_score for e in entities_list)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=entities_list,
            total_risk_score=total_risk,
            risk_level=risk_level,
            model_version="v2.2",
        )

    def _merge_and_arbitrate(
        self, rule_entities: List[Entity], bert_entities: List[Entity]
    ) -> List[Entity]:
        """
        合并规则引擎和 BERT 的结果，处理重叠。
        优先考虑高分和高置信度的实体。
        """
        all_entities = rule_entities + bert_entities
        if not all_entities:
            return []

        # 按得分、置信度和长度排序
        all_entities.sort(key=lambda x: (x.risk_score, x.confidence, len(x.text)), reverse=True)

        final_entities: List[Entity] = []

        for current in all_entities:
            is_redundant = False
            for existing in final_entities:
                # 使用字符偏移量进行更精确的重叠判断
                overlap = max(
                    0,
                    min(current.end_char, existing.end_char)
                    - max(current.start_char, existing.start_char),
                )
                if overlap > 0:
                    # 如果有重叠，且得分较低或置信度较低，则视为冗余
                    is_redundant = True
                    break

                # 保留原有的基于文本包含的逻辑作为补充
                if (current.text in existing.text or existing.text in current.text) and (
                    current.type == existing.type
                ):
                    is_redundant = True
                    break

            if not is_redundant:
                final_entities.append(current)

        return sorted(final_entities, key=lambda x: x.start_char)
