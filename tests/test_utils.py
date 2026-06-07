import pytest
from finance_risk_rag.utils import clean_text, split_text_by_sentence, calculate_risk_level

def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("3。5") == "3.5"
    assert clean_text("价格。") == "价格."
    assert clean_text("你好，世界；") == "你好,世界;"

def test_split_text_by_sentence():
    text = "这是一个测试。这是第二个测试！"
    # 设置一个较小的 max_len 强制其不合并，以便测试拆分
    sentences = split_text_by_sentence(text, max_len=10, min_len=1)
    assert len(sentences) == 2
    assert "这是一个测试。" in sentences
    assert "这是第二个测试！" in sentences

def test_calculate_risk_level():
    assert calculate_risk_level(20) == "低风险"
    assert calculate_risk_level(40) == "中风险"
    assert calculate_risk_level(70) == "高风险"
    assert calculate_risk_level(100) == "极高风险"
