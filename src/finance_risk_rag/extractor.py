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
    """实体得分策略接口"""

    @abstractmethod
    def calculate(self, entity_type: str, confidence: float, base_score: int) -> int:
        pass


class DefaultScoringStrategy(ScoringStrategy):
    """默认得分策略"""

    def calculate(self, entity_type: str, confidence: float, base_score: int) -> int:
        # 简单逻辑：基础分 * 置信度
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
                            start_char=start,
                            end_char=end,
                            context=context,
                            source="rule",
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

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _chunk_text(self, text: str, max_length: int = 510, overlap: int = 50) -> List[Tuple[str, int]]:
        """将长文本切分为带重叠的块"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_length, text_len)
            chunks.append((text[start:end], start))
            if end == text_len:
                break
            start += max_length - overlap
        return chunks

    def extract(self, text: str) -> List[Entity]:
        """提取实体，处理长文本"""
        if not self.is_available or not text:
            return []

        # 获取文本分块
        chunks = self._chunk_text(text)
        all_entities = []

        for chunk_text, offset in chunks:
            all_entities.extend(self._extract_segment(chunk_text, offset))

        # 精确去重：对于重叠区域可能提取到的相同实体，保留置信度高的
        unique_entities: Dict[Tuple[int, int, str], Entity] = {}
        for entity in all_entities:
            # 使用 (开始位置, 结束位置, 类型) 作为唯一标识
            key = (entity.start_char, entity.end_char, entity.type)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity

        return sorted(list(unique_entities.values()), key=lambda x: x.start_char)

    def _extract_segment(self, text: str, offset: int) -> List[Entity]:
        """提取单个片段的实体"""
        try:
            results = self.nlp(text)
            entities = []
            for res in results:
                start = res["start"] + offset
                end = res["end"] + offset
                entities.append(
                    Entity(
                        type=res["entity_group"],
                        text=res["word"],
                        risk_score=20,
                        confidence=float(res["score"]),
                        start_char=start,
                        end_char=end,
                        context=text[max(0, res["start"] - 40) : min(len(text), res["end"] + 40)],
                        source="bert",
                    )
                )
            return entities
        except Exception as e:
            logger.error(f"BERT segment extraction failed: {e}")
            return []


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

        # Advanced merging logic: Score-based arbitration and overlap resolution
        entities_list = self._merge_and_arbitrate(rule_entities, bert_entities)

        total_risk = sum(e.risk_score for e in entities_list)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=entities_list, total_risk_score=total_risk, risk_level=risk_level
        )

    def _merge_and_arbitrate(self, rule_entities: List[Entity], bert_entities: List[Entity]) -> List[Entity]:
        """
        合并规则引擎和 BERT 的结果，处理重叠。
        基于字符位置进行冲突仲裁，优先保留长实体和高置信度实体。
        """
        all_entities = rule_entities + bert_entities
        if not all_entities:
            return []

        # 排序：位置在前优先，相同位置长者优先，得分高者优先
        all_entities.sort(key=lambda x: (x.start_char, -(x.end_char - x.start_char), -x.risk_score))

        final_entities: List[Entity] = []

        for current in all_entities:
            if current.start_char == -1 or current.end_char == -1:
                # 降级处理没有偏移量的实体（如果有的话）
                final_entities.append(current)
                continue

            is_overlapped = False
            for existing in final_entities:
                if existing.start_char == -1:
                    continue

                # 检测重叠： [start1, end1] vs [start2, end2]
                if not (current.end_char <= existing.start_char or current.start_char >= existing.end_char):
                    # 发生重叠
                    # 仲裁：如果当前实体得分远高于已有实体，或者已有实体完全被当前包含
                    if current.risk_score > existing.risk_score * 1.5:
                        final_entities.remove(existing)
                        is_overlapped = False # 继续添加当前
                        break
                    else:
                        is_overlapped = True
                        break

            if not is_overlapped:
                final_entities.append(current)

        return sorted(final_entities, key=lambda x: x.start_char)
