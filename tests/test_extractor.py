import pytest
from pathlib import Path
from finance_risk_rag.extractor import RuleBasedExtractor, EntityMerger
from finance_risk_rag.models import Entity

def test_rule_based_extractor(tmp_path):
    rules_file = tmp_path / "rules.json"
    rules_file.write_text('{"liquidity_risk": {"keywords": ["现金"], "risk_score": 15}}', encoding="utf-8")

    extractor = RuleBasedExtractor(rules_file)
    text = "公司的现金储备不足。"
    entities = extractor.extract(text)

    assert len(entities) == 1
    assert entities[0].type == "liquidity_risk"
    assert entities[0].text == "现金"
    assert entities[0].risk_score == 15

def test_entity_merger():
    merger = EntityMerger()
    e1 = Entity(type="risk", text="abc", risk_score=10, confidence=0.8)
    e2 = Entity(type="risk", text="abc", risk_score=20, confidence=0.9)

    merged = merger.merge([e1], [e2])
    assert len(merged) == 1
    assert merged[0].risk_score == 20
    assert merged[0].confidence == 0.9
