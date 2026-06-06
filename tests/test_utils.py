import pytest
from src.finance_risk_rag.utils import clean_text, split_text_by_sentence, calculate_risk_level

def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("3。5") == "3.5"
    assert clean_text("你好。") == "你好."

def test_split_text_by_sentence():
    text = "这是第一个句子。这是第二个句子！"
    sentences = split_text_by_sentence(text, max_len=10)
    assert len(sentences) == 2
    assert "这是第一个句子。" in sentences
    assert "这是第二个句子！" in sentences

def test_calculate_risk_level():
    assert calculate_risk_level(10) == "低风险"
    assert calculate_risk_level(40) == "中风险"
    assert calculate_risk_level(70) == "高风险"
    assert calculate_risk_level(95) == "极高风险"
