"""
Risk entity extraction module.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import get_config
from .exceptions import RuleLoadError
from .models import Entity, ExtractionResult
from .utils import clean_text, calculate_risk_level, load_json_file


class RuleBasedExtractor:
    """Extracts risk entities using keyword matching."""

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        self.rules: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: Path) -> None:
        """Load entity extraction rules from JSON."""
        try:
            self.rules = load_json_file(rules_path)
            self.logger.info(f"Loaded {len(self.rules)} rule categories")
        except Exception as e:
            raise RuleLoadError(f"Failed to load rules: {e}")

    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text based on loaded rules."""
        if not text or not self.rules:
            return []

        entities = []
        for entity_type, config in self.rules.items():
            keywords = config.get("keywords", [])
            risk_score = config.get("risk_score", 10)

            for keyword in keywords:
                # Basic matching for CJK and non-CJK
                pattern = re.escape(keyword)
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start()
                    context = text[max(0, start-50):min(len(text), start+len(keyword)+50)]

                    entities.append(Entity(
                        type=entity_type,
                        text=keyword,
                        risk_score=risk_score,
                        confidence=1.0,
                        context=context.strip(),
                        source="rule"
                    ))
        return entities


class BERTExtractor:
    """Extracts risk entities using a BERT-based NER model."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None

        if model_path and model_path.exists():
            self._load_model()

    def _load_model(self):
        """Lazy load BERT model."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path))
            self.logger.info("BERT model loaded successfully")
        except ImportError:
            self.logger.warning("Transformers/Torch not installed. BERT extraction unavailable.")
        except Exception as e:
            self.logger.error(f"Failed to load BERT model: {e}")

    def extract(self, text: str) -> List[Entity]:
        """Placeholder for BERT extraction logic."""
        if not self.model:
            return []
        # Complex NER logic would go here
        return []


class EntityExtractionPipeline:
    """Pipeline for entity extraction and risk scoring."""

    def __init__(self, config=None) -> None:
        self.config = config or get_config()
        self.rule_extractor = RuleBasedExtractor(self.config.risk_entities_path)
        self.bert_extractor = BERTExtractor()  # Can be configured with path
        self.logger = logging.getLogger(__name__)

    def process(self, text: str) -> ExtractionResult:
        """Process text and return an extraction result."""
        cleaned = clean_text(text)

        # Combine results from both extractors
        rule_entities = self.rule_extractor.extract(cleaned)
        bert_entities = self.bert_extractor.extract(cleaned)

        combined_entities = rule_entities + bert_entities

        # Deduplicate
        unique_entities = {}
        for e in combined_entities:
            if e.key not in unique_entities:
                unique_entities[e.key] = e
            else:
                # Merge logic: keep the one with higher confidence/score
                existing = unique_entities[e.key]
                if e.confidence > existing.confidence:
                    unique_entities[e.key] = e

        final_entities = list(unique_entities.values())
        total_score = sum(e.risk_score for e in final_entities)
        risk_level = calculate_risk_level(total_score)

        return ExtractionResult(
            entities=final_entities,
            total_risk_score=total_score,
            risk_level=risk_level
        )
