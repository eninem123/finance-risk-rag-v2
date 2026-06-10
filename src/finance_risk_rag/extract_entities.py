"""
Finance-Risk-RAG 实体提取模块
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from finance_risk_rag.config import get_config
from finance_risk_rag.exceptions import ExtractionError, RuleLoadError
from finance_risk_rag.models import Entity, ExtractionResult
from finance_risk_rag.utils import (
    calculate_risk_level,
    clean_text,
    load_json_file,
    setup_logger,
)


class RuleBasedExtractor:
    def __init__(self, rules_path: Optional[Path] = None) -> None:
        self._rules: Dict[str, Any] = {}
        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: Path) -> None:
        try:
            self._rules = load_json_file(rules_path)
        except Exception as e:
            raise RuleLoadError(f"无法加载规则文件: {e}")

    def extract(self, text: str) -> List[Entity]:
        if not text or not self._rules:
            return []
        entities: List[Entity] = []
        for entity_type, config in self._rules.items():
            keywords = config.get("keywords", [])
            base_risk_score = config.get("risk_score", 10)
            for keyword in keywords:
                # 注意：CJK 文本不建议使用 \b
                pattern = re.escape(keyword)
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start()
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


class BERTExtractor:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Path) -> bool:
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForTokenClassification.from_pretrained(str(model_path))
            if self._model is not None:
                self._model.eval()
            return True
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def extract(self, text: str) -> List[Entity]:
        # 简化实现
        return []


class EntityMerger:
    def merge(self, rule_entities: List[Entity], bert_entities: List[Entity]) -> List[Entity]:
        merged: Dict[Tuple[str, str], Entity] = {}
        for entity in rule_entities + bert_entities:
            key = entity.key
            if key not in merged:
                merged[key] = entity
            else:
                existing = merged[key]
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.risk_score = max(existing.risk_score, entity.risk_score)
        return list(merged.values())


class EntityExtractionPipeline:
    def __init__(self, config: Optional[Any] = None) -> None:
        self._config = config or get_config()
        self._logger = setup_logger("entity_extraction")
        self._rule_extractor = RuleBasedExtractor()
        self._bert_extractor = BERTExtractor()
        self._merger = EntityMerger()

    def initialize(self) -> None:
        if self._config.risk_entities_path.exists():
            self._rule_extractor.load_rules(self._config.risk_entities_path)
        if self._config.bert_local_path:
            self._bert_extractor.load_model(self._config.bert_local_path)

    def process(self, text_path: Path) -> ExtractionResult:
        if not text_path.exists():
            raise ExtractionError(f"文件不存在: {text_path}")
        text = clean_text(text_path.read_text(encoding="utf-8"))
        rule_entities = self._rule_extractor.extract(text)
        bert_entities = (
            self._bert_extractor.extract(text) if self._bert_extractor.is_available else []
        )
        final_entities = self._merger.merge(rule_entities, bert_entities)
        total_risk = sum(e.risk_score for e in final_entities)
        return ExtractionResult(
            entities=final_entities,
            total_risk_score=total_risk,
            risk_level=calculate_risk_level(total_risk),
        )
