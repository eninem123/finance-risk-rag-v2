"""
Finance-Risk-RAG 实体提取管道
============================
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .utils import calculate_risk_level, clean_text, load_json_file, setup_logger

logger = setup_logger("extractor", "logs/extractor_optimized.log")


@dataclass
class Entity:
    type: str
    text: str
    risk_score: int
    context: str


class EntityExtractor:
    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config or get_config()
        self.rules = load_json_file(self.config.risk_entities_path)

    def extract_from_text(self, text: str) -> List[Entity]:
        text = clean_text(text)
        entities = []

        for e_type, rule in self.rules.items():
            keywords = rule.get("keywords", [])
            score = rule.get("risk_score", 10)

            for kw in keywords:
                # Use word boundary only for alphanumeric keywords (mostly English)
                # Chinese characters do not use \b boundaries
                pattern = re.escape(kw)
                if re.match(r"^[a-zA-Z0-9]", kw):
                    pattern = r"\b" + pattern
                if re.search(r"[a-zA-Z0-9]$", kw):
                    pattern = pattern + r"\b"

                for match in re.finditer(pattern, text, re.I):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    entities.append(
                        Entity(type=e_type, text=kw, risk_score=score, context=text[start:end])
                    )

        return entities


class ExtractionPipeline:
    def __init__(self) -> None:
        self.extractor = EntityExtractor()

    def run(self, input_path: Path) -> Dict[str, Any]:
        text = input_path.read_text(encoding="utf-8")
        entities = self.extractor.extract_from_text(text)

        total_score = sum(e.risk_score for e in entities)
        risk_level = calculate_risk_level(total_score)

        return {
            "entities_count": len(entities),
            "total_risk_score": total_score,
            "risk_level": risk_level,
            "entities": [
                {"type": e.type, "text": e.text, "score": e.risk_score} for e in entities[:10]
            ],
        }
