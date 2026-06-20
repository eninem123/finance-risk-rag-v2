import json

from src.finance_risk_rag.extractor import RuleBasedExtractor


def test_rule_based_extractor(tmp_path):
    # Create dummy rules
    rules_file = tmp_path / "rules.json"

    rules = {"liquidity_risk": {"keywords": ["现金流", "流动性"], "risk_score": 20}}
    rules_file.write_text(json.dumps(rules), encoding="utf-8")

    extractor = RuleBasedExtractor(rules_path=rules_file)
    text = "该公司的现金流存在明显压力，流动性不足。"
    entities = extractor.extract(text)

    assert len(entities) == 2
    assert entities[0].type == "liquidity_risk"
    assert any(e.text == "现金流" for e in entities)
    assert any(e.text == "流动性" for e in entities)

    # Check offsets
    for e in entities:
        assert text[e.start_char : e.end_char] == e.text
