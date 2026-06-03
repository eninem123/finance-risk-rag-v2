"""
Finance-Risk-RAG Extractor Module
=================================

Extracts risk entities from financial text using rules and AI.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import get_config
from .utils import load_json, save_json, setup_logger

@dataclass
class RiskEntity:
    type: str
    text: str
    risk_score: int
    context: str

class EntityExtractor:
    """Extracts risk entities from text."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("EntityExtractor")
        self.rules = load_json(self.config.risk_entities_path)

    def extract(self, text: str) -> List[RiskEntity]:
        """Extract entities based on configured rules."""
        entities = []

        for category, config in self.rules.items():
            keywords = config.get("keywords", [])
            score = config.get("risk_score", 10)

            for kw in keywords:
                # Use regex for better matching
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                for match in pattern.finditer(text):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].replace('\n', ' ')

                    entities.append(RiskEntity(
                        type=category,
                        text=kw,
                        risk_score=score,
                        context=f"...{context}..."
                    ))

        return entities

    def summarize_risk(self, entities: List[RiskEntity]) -> Dict[str, Any]:
        """Summarize total risk based on extracted entities."""
        total_score = sum(e.risk_score for e in entities)
        return {
            "total_score": total_score,
            "entity_count": len(entities),
            "entities": [asdict(e) for e in entities],
            "top_risks": [asdict(e) for e in sorted(entities, key=lambda x: x.risk_score, reverse=True)[:5]]
        }

    def save_results(self, summary: Dict[str, Any], output_path: Path):
        """Save summary to JSON."""
        save_json(summary, output_path)
        self.logger.info(f"Results saved to {output_path}")
