"""
Finance-Risk-RAG 工具模块单元测试
"""

from finance_risk_rag.utils import clean_text, split_text_by_sentence, calculate_risk_level


def test_clean_text():
    text = "  Hello \n\n World! 3。5 "
    cleaned = clean_text(text)
    assert cleaned == "Hello World! 3.5"

    chinese_text = "测试。测试。"
    assert clean_text(chinese_text) == "测试.测试."


def test_split_text_by_sentence():
    text = "这是第一个句子。这是第二个句子。这是第三个句子。"
    sentences = split_text_by_sentence(text, max_len=10, min_len=1)
    assert len(sentences) >= 3
    for s in sentences:
        assert len(s) > 0


def test_calculate_risk_level():
    assert calculate_risk_level(20) == "低风险"
    assert calculate_risk_level(50) == "中风险"
    assert calculate_risk_level(80) == "高风险"
    assert calculate_risk_level(100) == "极高风险"
