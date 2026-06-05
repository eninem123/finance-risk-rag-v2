"""
Unit tests for utility functions.
"""

from src.finance_risk_rag.utils import (
    calculate_risk_level,
    clean_text,
    split_text_by_sentence,
)


def test_clean_text():
    assert clean_text("  Hello   World  ") == "Hello World"
    assert clean_text("测试，；。") == "测试,;."


def test_split_text_by_sentence():
    text = "This is a sentence. This is another! And a third?"
    sentences = split_text_by_sentence(text, max_len=20)
    assert len(sentences) == 3
    assert sentences[0] == "This is a sentence."


def test_calculate_risk_level():
    assert calculate_risk_level(20) == "Low Risk"
    assert calculate_risk_level(50) == "Medium Risk"
    assert calculate_risk_level(80) == "High Risk"
    assert calculate_risk_level(100) == "Extreme Risk"
