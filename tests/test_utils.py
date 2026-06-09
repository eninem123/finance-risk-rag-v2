import pytest
from finance_risk_rag.utils import clean_text, split_text_by_sentence, calculate_risk_level

def test_clean_text():
    text = "  Hello   World! 。 1。5  "
    cleaned = clean_text(text)
    assert cleaned == "Hello World! . 1.5"

def test_split_text_by_sentence():
    text = "第一句。第二句！第三句？"
    # 设置极小的 max_len 以防止合并，从而测试拆分逻辑
    sentences = split_text_by_sentence(text, max_len=5, min_len=1)
    assert len(sentences) == 3
    assert "第一句。" in sentences
    assert "第二句！" in sentences

def test_calculate_risk_level():
    assert calculate_risk_level(20) == "低风险"
    assert calculate_risk_level(45) == "中风险"
    assert calculate_risk_level(75) == "高风险"
    assert calculate_risk_level(95) == "极高风险"
