"""
Finance-Risk-RAG 实体提取模块
============================
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .config import get_config
from .exceptions import ExtractionError
from .models import Entity, ExtractionResult
from .utils import calculate_risk_level, clean_text, load_json_file

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """评分策略抽象基类"""

    @abstractmethod
    def calculate_score(self, entity_type: str, confidence: float, base_score: int) -> int:
        """根据实体类型、置信度和基础分数计算最终风险分数"""
        pass


class DefaultScoringStrategy(ScoringStrategy):
    """默认评分策略"""

    def calculate_score(self, entity_type: str, confidence: float, base_score: int) -> int:
        # 基础逻辑：分数 = 基础分数 * 置信度 (向上取整)
        return int(base_score * confidence + 0.5)


class BaseExtractor(ABC):
    """提取器基类"""

    def __init__(self, scoring_strategy: Optional[ScoringStrategy] = None):
        self.scoring_strategy = scoring_strategy or DefaultScoringStrategy()

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """从文本中提取实体"""
        pass


class RuleBasedExtractor(BaseExtractor):
    """基于规则的实体提取器"""

    def __init__(
        self,
        config=None,
        rules_path: Optional[Path] = None,
        scoring_strategy: Optional[ScoringStrategy] = None,
    ):
        super().__init__(scoring_strategy)
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

                    confidence = 1.0
                    risk_score = self.scoring_strategy.calculate_score(
                        entity_type, confidence, base_risk_score
                    )

                    entities.append(
                        Entity(
                            type=entity_type,
                            text=keyword,
                            risk_score=risk_score,
                            confidence=confidence,
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
        super().__init__(scoring_strategy)
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

    def _chunk_text(self, text: str, max_length: int = 510, overlap: int = 50) -> List[Tuple[str, int]]:
        """
        将文本切分为适合 BERT 的块，带有重叠。
        返回 (块文本, 原始文本中的起始偏移) 的列表。
        """
        # 简单按字符切分，实际中可能需要按 token 切分更精确
        # 这里使用保守的字符长度估计 (1 token ≈ 1-2 chars for ZH)
        # 实际上 BERT-Chinese 通常是按字符编码的
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append((text[start:end], start))
            if end == len(text):
                break
            start += max_length - overlap
        return chunks

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available() or not text:
            return []

        try:
            # 处理长文本：分块
            text_chunks = self._chunk_text(text)
            all_entities = []
            seen_keys = set()

            for chunk_text, offset in text_chunks:
                results = self.nlp(chunk_text)
                for res in results:
                    entity_type = res["entity_group"]
                    confidence = float(res["score"])
                    text_val = res["word"]
                    start_char = res["start"] + offset
                    end_char = res["end"] + offset

                    # 去重（对于重叠部分的重复识别）
                    key = (entity_type, text_val, start_char)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    # Default risk score for BERT entities is 20
                    risk_score = self.scoring_strategy.calculate_score(entity_type, confidence, 20)

                    all_entities.append(
                        Entity(
                            type=entity_type,
                            text=text_val,
                            risk_score=risk_score,
                            confidence=confidence,
                            context=text[max(0, start_char - 40) : min(len(text), end_char + 40)],
                            source="bert",
                            start_char=start_char,
                            end_char=end_char,
                        )
                    )
            return all_entities
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
        scoring_strategy=None,
    ):
        self.config = config or get_config()
        self.scoring_strategy = scoring_strategy or DefaultScoringStrategy()
        self.rule_extractor = rule_extractor or RuleBasedExtractor(
            config=self.config, scoring_strategy=self.scoring_strategy
        )
        self.bert_extractor = bert_extractor or BERTExtractor(
            model_path=self.config.bert_local_path, scoring_strategy=self.scoring_strategy
        )

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
        优先考虑高分和高置信度的实体。
        """
        all_entities = rule_entities + bert_entities
        if not all_entities:
            return []

        # 按得分和置信度排序
        all_entities.sort(key=lambda x: (x.risk_score, x.confidence), reverse=True)

        final_entities: List[Entity] = []

        for current in all_entities:
            is_redundant = False
            for existing in final_entities:
                # 使用字符偏移量进行精确的重叠检测
                overlap = max(current.start_char, existing.start_char) < min(
                    current.end_char, existing.end_char
                )
                if overlap:
                    is_redundant = True
                    break

            if not is_redundant:
                final_entities.append(current)

        return final_entities
