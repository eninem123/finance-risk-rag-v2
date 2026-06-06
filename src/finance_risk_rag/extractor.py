"""
Finance-Risk-RAG 实体提取模块
============================
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .config import get_config
from .exceptions import ExtractionError
from .models import Entity, ExtractionResult
from .utils import calculate_risk_level, clean_text, load_json_file

logger = logging.getLogger(__name__)


class RuleBasedExtractor:
    """基于规则的实体提取器"""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules = {}
        if rules_path:
            self.load_rules(rules_path)
        else:
            config = get_config()
            if config.risk_entities_path.exists():
                self.load_rules(config.risk_entities_path)

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
                    key = (entity_type, keyword, start)
                    if key in seen:
                        continue
                    seen.add(key)

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

    def extract(self, text: str) -> List[Entity]:
        if not self.is_available or not text:
            return []

        try:
            results = self.nlp(text)
            entities = []
            for res in results:
                entities.append(
                    Entity(
                        type=res["entity_group"],
                        text=res["word"],
                        risk_score=20,  # Default risk score for BERT entities
                        confidence=float(res["score"]),
                        context=text[max(0, res["start"] - 40) : min(len(text), res["end"] + 40)],
                        source="bert",
                    )
                )
            return entities
        except Exception as e:
            logger.error(f"BERT extraction failed: {e}")
            return []


class EntityExtractionPipeline:
    """实体提取管道"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.rule_extractor = RuleBasedExtractor()
        self.bert_extractor = BERTExtractor(self.config.bert_local_path)

    def process(self, text_or_path: Union[str, Path]) -> ExtractionResult:
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            text = text_or_path

        text = clean_text(text)

        rule_entities = self.rule_extractor.extract(text)
        bert_entities = self.bert_extractor.extract(text)

        # Simple merging logic
        final_entities = rule_entities + bert_entities
        # Deduplicate
        unique_entities = {}
        for e in final_entities:
            key = e.key
            if key not in unique_entities or e.risk_score > unique_entities[key].risk_score:
                unique_entities[key] = e

        entities_list = list(unique_entities.values())
        total_risk = sum(e.risk_score for e in entities_list)
        risk_level = calculate_risk_level(total_risk)

        return ExtractionResult(
            entities=entities_list, total_risk_score=total_risk, risk_level=risk_level
        )
