"""
Unit tests for entity extraction.
"""

from src.finance_risk_rag.extractor import RuleBasedExtractor
from src.finance_risk_rag.models import Entity


def test_rule_based_extractor_extract():
    extractor = RuleBasedExtractor()
    extractor.rules = {
        "liquidity_risk": {"keywords": ["cash reserve", "liquidity"], "risk_score": 20}
    }

    text = "The company has a low cash reserve and faces liquidity issues."
    entities = extractor.extract(text)

    assert len(entities) == 2
    assert any(e.text == "cash reserve" for e in entities)
    assert any(e.text == "liquidity" for e in entities)
    assert all(e.type == "liquidity_risk" for e in entities)
